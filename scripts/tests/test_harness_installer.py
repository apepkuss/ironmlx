"""Exercise install decisions without changing /Applications or user Harness state."""
import importlib.util
import json
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('installer_package', ROOT / 'scripts/package-installer.py')
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)
POLICY = ROOT / 'ironmlx-app/Packaging/Installer/distribution.js'


@unittest.skipUnless(shutil.which('node'), 'Node is needed to exercise Installer JavaScript')
class InstallPolicyTests(unittest.TestCase):
    def decide(self, installed, located=None):
        # The real Installer APIs are covered separately below; this matrix exercises their inputs.
        source = '''const vm = require('node:vm');
let input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
let context = {
 my: {target: {mountpoint: '/'}, search: {results: {existingHarness: input.located || []}}},
 system: {files: {
  bundleAtPath: p => input.installed[p] === false ? null : input.installed[p],
  fileExistsAtPath: p => Object.hasOwn(input.installed, p)
 }}
};
vm.createContext(context); vm.runInContext(input.policy, context);
process.stdout.write(JSON.stringify({state: context.harnessState(), install: context.shouldInstallHarness()}));'''
        data = dict(installed=installed, located=located or [], policy=POLICY.read_text())
        return json.loads(subprocess.check_output(['node', '-e', source], input=json.dumps(data), text=True))

    def test_install_reuse_upgrade_and_unknown_versions(self):
        path = '/Applications/DeepSeek Harness.app'
        self.assertEqual(self.decide({}), dict(state='install', install=True))
        for version, state in [('0.1.9', 'upgrade'), ('0.2.1-rc.1', 'upgrade'),
                               ('0.2.1-rc.2', 'reuse'), ('0.2.1', 'reuse'), ('1.0.0', 'reuse'),
                               ('unknown', 'blocked'), ('0.2.1-alpha.1', 'blocked')]:
            with self.subTest(version=version):
                value = self.decide({path: dict(CFBundleIdentifier='com.deepseek.dsh', CFBundleShortVersionString=version)})
                self.assertEqual(value, dict(state=state, install=state == 'upgrade'))
        for info in [False, {}, dict(CFBundleIdentifier='io.dsh.desktop', CFBundleShortVersionString='0.12.1')]:
            self.assertEqual(self.decide({path: info})['state'], 'blocked')

    def test_moved_and_user_installs_are_reused_without_relocation(self):
        for path in ['/Users/test/Applications/Harness.app', '/Applications/Tools/Harness.app']:
            info = dict(CFBundleIdentifier='com.deepseek.dsh', CFBundleShortVersionString='0.2.1-rc.2')
            self.assertEqual(self.decide({path: info}, [path]), dict(state='reuse', install=False))
            info['CFBundleShortVersionString'] = '0.1.0'
            self.assertEqual(self.decide({path: info}, [path]), dict(state='blocked', install=False))

    def test_xml_keeps_required_app_and_user_control_of_optional_component(self):
        tree = ET.fromstring(package.distribution('0.2.0'))
        choices = {row.attrib['id']: row.attrib for row in tree.findall('choice')}
        self.assertEqual(choices['ironmlx']['enabled'], 'false')
        self.assertEqual(choices['ironmlx']['selected'], 'true')
        self.assertNotIn('selected', choices['harness'])
        self.assertEqual(choices['harness']['start_selected'], 'true')
        active = tree.find("pkg-ref[@active]")
        self.assertEqual(active.attrib['active'], 'shouldInstallHarness()')
        self.assertNotIn('allow-external-scripts', tree.find('options').attrib)


@unittest.skipUnless(shutil.which('pkgbuild'), 'requires macOS packaging tools')
class NativeInstallerTests(unittest.TestCase):
    def test_component_payloads_choices_and_opt_out_with_real_installer(self):
        with tempfile.TemporaryDirectory(prefix='ironmlx-installer-test-') as temporary:
            root = Path(temporary)
            apps = []
            for name, identifier in [('IronMLX', 'com.ironmlx.app'), ('DeepSeek Harness', 'com.deepseek.dsh')]:
                app = root / (name + '.app')
                (app / 'Contents').mkdir(parents=True)
                (app / 'Contents/Info.plist').write_bytes(plistlib.dumps(dict(
                    CFBundleIdentifier=identifier, CFBundleShortVersionString='0.2.0', CFBundleVersion='1',
                    CFBundlePackageType='APPL')))
                (app / 'Contents/sentinel.txt').write_text('payload must remain exact')
                apps.append(app)
            output = root / 'IronMLX.pkg'
            with patch.object(package, 'validate_inputs', return_value='0.2.0'):
                package.assemble(*apps, output)
                # Verify detects changed payloads even when the container remains structurally valid.
                (apps[0] / 'Contents/sentinel.txt').write_text('changed reference')
                with self.assertRaisesRegex(ValueError, 'payload differs'):
                    package.verify(*apps, output)
            raw = subprocess.check_output(['installer', '-showChoicesXML', '-pkg', output, '-target', '/'])
            choices = plistlib.loads(raw)[0]['childItems']
            self.assertEqual([(c['choiceIdentifier'], c['choiceIsSelected'], c['choiceIsEnabled']) for c in choices],
                             [('ironmlx', 1, False), ('harness', 1, True)])
            self.assertNotIn('harnessDescription()', choices[1]['choiceDescription'])
            changes = root / 'choices.plist'
            changes.write_bytes(plistlib.dumps([dict(choiceIdentifier='harness', choiceAttribute='selected', attributeSetting=0)]))
            raw = subprocess.check_output(['installer', '-showChoicesAfterApplyingChangesXML', changes,
                                           '-pkg', output, '-target', '/'])
            # This only evaluates choices: no privileged install and no system/user file mutations.
            evaluated = plistlib.loads(raw)
            if evaluated and 'childItems' in evaluated[0]:
                self.assertEqual(evaluated[0]['childItems'][1]['choiceIsSelected'], 0)
            else:
                self.assertTrue(any(c.get('choiceIdentifier') == 'harness' and
                                    c.get('choiceAttribute') == 'selected' and c.get('attributeSetting') == 0
                                    for c in evaluated))

    def test_native_installer_evaluates_real_files_for_each_install_state(self):
        with tempfile.TemporaryDirectory(prefix='ironmlx-native-choice-') as temporary:
            root = Path(temporary)
            resources = root / 'resources'
            shutil.copytree(package.RESOURCES, resources)
            target = root / 'installed/DeepSeek Harness.app'
            policy = (resources / 'distribution.js').read_text().replace(
                "'/Applications/DeepSeek Harness.app'", json.dumps(str(target))).replace(
                'var located = my.search.results.existingHarness;', 'var located = [];')
            (resources / 'distribution.js').write_text(policy)
            payload = root / 'empty'
            payload.mkdir()
            for component in ['IronMLX-component.pkg', 'Harness-component.pkg']:
                subprocess.run(['pkgbuild', '--root', payload, '--identifier', 'com.ironmlx.test.' + component,
                                '--version', '1', root / component], check=True, stdout=subprocess.DEVNULL)
            with patch.object(package, 'RESOURCES', resources):
                (root / 'Distribution').write_text(package.distribution('0.2.0'))
            output = root / 'product.pkg'
            subprocess.run(['productbuild', '--distribution', root / 'Distribution', '--resources', resources,
                            '--package-path', root, output], check=True, stdout=subprocess.DEVNULL)
            def selected():
                raw = subprocess.check_output(['installer', '-showChoicesXML', '-pkg', output, '-target', '/'])
                return plistlib.loads(raw)[0]['childItems'][1]
            self.assertTrue(selected()['pathsOfActivePackagesInChoice'])
            (target / 'Contents').mkdir(parents=True)
            for version, active in [('0.1.0', True), ('0.2.1-rc.2', False), ('1.0.0', False), ('unknown', False)]:
                (target / 'Contents/Info.plist').write_bytes(plistlib.dumps(dict(
                    CFBundleIdentifier='com.deepseek.dsh', CFBundleShortVersionString=version)))
                self.assertEqual(bool(selected()['pathsOfActivePackagesInChoice']), active)
            shutil.rmtree(target)
            changes = root / 'choices.plist'
            changes.write_bytes(plistlib.dumps([dict(choiceIdentifier='harness', choiceAttribute='selected', attributeSetting=0)]))
            raw = subprocess.check_output(['installer', '-showChoicesAfterApplyingChangesXML', changes,
                                           '-pkg', output, '-target', '/'])
            evaluated = plistlib.loads(raw)
            self.assertTrue(any(c.get('choiceIdentifier') == 'harness' and
                                c.get('choiceAttribute') == 'selected' and c.get('attributeSetting') == 0
                                for c in evaluated))

    def test_destination_recheck_refuses_newer_unknown_unrelated_and_symlink_targets(self):
        script = ROOT / 'ironmlx-app/Packaging/Installer/harness-preinstall'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'Applications/DeepSeek Harness.app'
            (root / 'Applications').mkdir()
            def run():
                return subprocess.run(['bash', script, 'unused', 'unused', root], capture_output=True).returncode
            self.assertEqual(run(), 0)
            (target / 'Contents').mkdir(parents=True)
            for version, expected in [('0.1.0', 0), ('0.2.1-rc.1', 0), ('0.2.1-rc.2', 1),
                                      ('0.2.1', 1), ('0.3.0', 1), ('unknown', 1)]:
                (target / 'Contents/Info.plist').write_bytes(plistlib.dumps(dict(
                    CFBundleIdentifier='com.deepseek.dsh', CFBundleShortVersionString=version)))
                self.assertEqual(run(), expected)
            (target / 'Contents/Info.plist').write_bytes(plistlib.dumps(dict(
                CFBundleIdentifier='io.dsh.desktop', CFBundleShortVersionString='0.1.0')))
            self.assertNotEqual(run(), 0)
            shutil.rmtree(target)
            target.symlink_to(root / 'outside')
            self.assertNotEqual(run(), 0)


class OfficialSourceTests(unittest.TestCase):
    def test_bad_checksum_and_escaping_symlink_fail_before_extraction(self):
        import hashlib
        import stat
        import zipfile
        spec = importlib.util.spec_from_file_location('harness_source', ROOT / 'scripts/prepare-harness-desktop.py')
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'harness.zip'
            lock = root / 'source.json'
            metadata = json.loads((ROOT / 'scripts/harness-desktop-source.json').read_text())
            with zipfile.ZipFile(archive, 'w') as stream:
                entry = zipfile.ZipInfo('DeepSeek Harness.app/escape')
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                stream.writestr(entry, '../../outside')
            lock.write_text(json.dumps(metadata))
            with patch.object(source, 'SOURCE', lock), patch.object(source.subprocess, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    source.prepare(root / 'result.app', archive)
                run.assert_not_called()
                metadata['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
                lock.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'escaping Harness symlink'):
                    source.prepare(root / 'result.app', archive)
                run.assert_not_called()
                self.assertFalse((root / 'result.app').exists())


if __name__ == '__main__':
    unittest.main()
