#!/usr/bin/env python3
"""Build/verify a two-component macOS installer. App-only updates remain independent."""
import argparse
import importlib.util
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / 'ironmlx-app/Packaging/Installer'
SOURCE = json.loads((ROOT / 'scripts/harness-desktop-source.json').read_text())
spec = importlib.util.spec_from_file_location('release_archives', ROOT / 'scripts/release-archives.py')
archives = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archives)
require = archives.require


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True, stdout=subprocess.DEVNULL)


def installer_script():
    provenance = json.dumps(SOURCE, sort_keys=True, indent=2)
    license_text = (RESOURCES / 'DeepSeek-Harness-LICENSE.txt').read_text()
    return '/* Optional official component provenance and license:\n' + provenance + '\n' + license_text + '*/\n' + (RESOURCES / 'distribution.js').read_text()


def distribution(version):
    # active is a second guard: selecting Harness cannot replace a compatible/newer copy.
    return f'''<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>IronMLX</title>
  <options customize="always" require-scripts="true" hostArchitectures="arm64"/>
  <domains enable_localSystem="true" enable_currentUserHome="false" enable_anywhere="false"/>
  <volume-check script="startupVolume()"><allowed-os-versions><os-version min="26.4"/></allowed-os-versions></volume-check>
  <choices-outline><line choice="ironmlx"/><line choice="harness"/></choices-outline>
  <choice id="ironmlx" title="IronMLX" description="ironmlx.description" selected="true" enabled="false">
    <pkg-ref id="com.ironmlx.installer.app"/>
  </choice>
  <choice id="harness" title="DeepSeek Harness Desktop" description="harness.install" start_selected="true">
    <pkg-ref id="com.ironmlx.installer.harness"/>
  </choice>
  <locator><search id="existingHarness" type="component"><bundle id="com.deepseek.dsh" path="Applications/DeepSeek Harness.app"/></search></locator>
  <script><![CDATA[
{installer_script()}
  ]]></script>
  <pkg-ref id="com.ironmlx.installer.app" version="{version}" auth="root">IronMLX-component.pkg</pkg-ref>
  <pkg-ref id="com.ironmlx.installer.app"><must-close><app id="com.ironmlx.app"/></must-close></pkg-ref>
  <pkg-ref id="com.ironmlx.installer.harness" version="{SOURCE['version']}" auth="root" active="shouldInstallHarness()">Harness-component.pkg</pkg-ref>
  <pkg-ref id="com.ironmlx.installer.harness"><relocate search-id="existingHarness"><bundle id="com.deepseek.dsh"/></relocate><must-close><app id="com.deepseek.dsh"/></must-close></pkg-ref>
</installer-gui-script>
'''


def validate_inputs(app, harness):
    with (app / 'Contents/Info.plist').open('rb') as stream:
        info = plistlib.load(stream)
    version = (ROOT / 'VERSION').read_text().strip()
    require(info.get('CFBundleIdentifier') == 'com.ironmlx.app', 'unexpected IronMLX app identity')
    require(info.get('CFBundleShortVersionString') == version, 'IronMLX version differs')
    run('python3', ROOT / 'scripts/prepare-harness-desktop.py', harness, '--verify-only')
    return version


def component(app, identifier, version, output, staging, scripts=None):
    # Disable automatic bundle relocation: never overwrite a different installed copy.
    root = staging / identifier
    applications = root / 'Applications'
    applications.mkdir(parents=True)
    run('ditto', app, applications / app.name)
    manifest = staging / (identifier + '.plist')
    run('pkgbuild', '--analyze', '--root', root, manifest)
    rows = plistlib.loads(manifest.read_bytes())
    # The official app contains nested runtime applications; the entire outer app is one unit.
    rows = [row for row in rows if row['RootRelativeBundlePath'] == 'Applications/' + app.name]
    require(len(rows) == 1, 'expected one outer application component')
    rows[0].update(BundleIsRelocatable=False, BundleIsVersionChecked=False, BundleOverwriteAction='upgrade')
    rows[0].pop('ChildBundles', None)
    manifest.write_bytes(plistlib.dumps(rows))
    run('pkgbuild', '--root', root, '--component-plist', manifest, '--identifier', identifier,
        '--version', version, '--install-location', '/', '--ownership', 'recommended',
        *(['--scripts', scripts] if scripts else []), output)


def assemble(app, harness, output):
    version = validate_inputs(app, harness)
    require(not output.exists(), 'installer output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ironmlx-installer-', dir=output.parent) as temporary:
        staging = Path(temporary)
        component(app, 'com.ironmlx.installer.app', version, staging / 'IronMLX-component.pkg', staging)
        scripts = staging / 'harness-scripts'
        scripts.mkdir()
        shutil.copy2(RESOURCES / 'harness-preinstall', scripts / 'preinstall')
        component(harness, 'com.ironmlx.installer.harness', SOURCE['version'],
                  staging / 'Harness-component.pkg', staging, scripts)
        definition = staging / 'Distribution'
        definition.write_text(distribution(version))
        resources = staging / 'resources'
        shutil.copytree(RESOURCES, resources)
        run('productbuild', '--distribution', definition, '--resources', resources,
            '--package-path', staging, output)
    verify(app, harness, output)


def verify(app, harness, package):
    version = validate_inputs(app, harness)
    with tempfile.TemporaryDirectory(prefix='ironmlx-installer-verify-') as temporary:
        expanded = Path(temporary) / 'expanded'
        run('pkgutil', '--expand-full', package, expanded)
        definition = expanded / 'Distribution'
        tree = ET.parse(definition).getroot()
        require({p.name for p in expanded.iterdir()} == {
            'Distribution', 'Resources', 'IronMLX-component.pkg', 'Harness-component.pkg'}, 'unexpected installer entries')
        require(tree.find('options').attrib.get('customize') == 'always', 'component selection must be visible')
        require(tree.find('options').attrib.get('hostArchitectures') == 'arm64', 'unexpected installer architecture')
        require(tree.find('volume-check').attrib.get('script') == 'startupVolume()', 'startup volume guard missing')
        active = tree.findall('pkg-ref[@active]')
        require(len(active) == 1 and active[0].attrib['id'] == 'com.ironmlx.installer.harness'
                and active[0].attrib['active'] == 'shouldInstallHarness()', 'conditional Harness install guard missing')
        references = {row.attrib['id'] for row in tree.findall('pkg-ref')}
        require(references == {'com.ironmlx.installer.app', 'com.ironmlx.installer.harness'}, 'unexpected component references')
        choices = {c.attrib['id']: c.attrib for c in tree.findall('choice')}
        require(set(choices) == {'ironmlx', 'harness'}, 'unexpected installer choices')
        require(choices['ironmlx'].get('selected') == 'true' and choices['ironmlx'].get('enabled') == 'false',
                'IronMLX must be mandatory')
        require(choices['harness'].get('start_selected') == 'true' and 'selected' not in choices['harness'],
                'Harness must be optional and initially selected')
        require(tree.find('script').text.strip() == installer_script().strip(),
                'installer detection policy differs')
        for name, reference in [('IronMLX-component.pkg', app), ('Harness-component.pkg', harness)]:
            root = expanded / name
            require(archives.inventory(root / 'Payload/Applications' / reference.name) == archives.inventory(reference),
                    f'installer payload differs: {name}')
            info = ET.parse(root / 'PackageInfo').getroot()
            require(info.attrib.get('identifier') == ('com.ironmlx.installer.app' if name.startswith('IronMLX')
                    else 'com.ironmlx.installer.harness'), 'unexpected component identity')
            require(info.attrib.get('install-location') == '/', 'unexpected installation destination')
            payload = root / 'Payload'
            require({p.name for p in payload.iterdir()} == {'Applications'}, 'unexpected payload root')
            require({p.name for p in (payload / 'Applications').iterdir()} == {reference.name}, 'unexpected app payload')
        preinstall = expanded / 'Harness-component.pkg/Scripts/preinstall'
        require({p.name for p in preinstall.parent.iterdir()} == {'preinstall'}, 'unexpected Harness install scripts')
        require(preinstall.read_bytes() == (RESOURCES / 'harness-preinstall').read_bytes(), 'missing destination recheck')
        require(not (expanded / 'IronMLX-component.pkg/Scripts').exists(), 'unexpected IronMLX install scripts')
        for locale in ('en', 'zh-Hans', 'zh-Hant', 'ja', 'ko'):
            require((expanded / f'Resources/{locale}.lproj/Localizable.strings').read_bytes() ==
                    (RESOURCES / f'{locale}.lproj/Localizable.strings').read_bytes(), 'installer localization differs')
    print(f'Installer content verified (signing is a separate gate): {package}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('assemble', 'verify'))
    parser.add_argument('app', type=Path)
    parser.add_argument('harness', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    function = assemble if args.action == 'assemble' else verify
    function(args.app.resolve(), args.harness.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
