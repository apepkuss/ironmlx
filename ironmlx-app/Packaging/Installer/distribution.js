// Evaluated by macOS Installer. No configuration files are created at install time.
var harnessMinimum = '0.2.1-rc.2';
var harnessBundleID = 'com.deepseek.dsh';
function versionParts(value) {
    var match = /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-rc\.(0|[1-9][0-9]*))?$/.exec(value || '');
    return match ? [Number(match[1]), Number(match[2]), Number(match[3]), match[4] === undefined ? Infinity : Number(match[4])] : null;
}
function compareVersion(a, b) {
    for (var i = 0; i < a.length; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}
function harnessState() {
    var target = '/Applications/DeepSeek Harness.app';
    var paths = [target];
    var located = my.search.results.existingHarness;
    if (typeof located === 'string' && located) paths.push(located);
    else if (located && located.length) paths = paths.concat(located);
    var oldTarget = false, blocked = false;
    for (var i = 0; i < paths.length; i++) {
        var info = system.files.bundleAtPath(paths[i]);
        if (!info) {
            if (paths[i] === target && system.files.fileExistsAtPath(target)) blocked = true;
            continue;
        }
        if (info.CFBundleIdentifier !== harnessBundleID) {
            if (paths[i] === target) blocked = true;
            continue;
        }
        var version = versionParts(info.CFBundleShortVersionString);
        if (!version) { blocked = true; continue; }
        if (compareVersion(version, versionParts(harnessMinimum)) >= 0) return 'reuse';
        if (paths[i] === target) oldTarget = true;
        else blocked = true; // Never create a second copy beside an incompatible moved installation.
    }
    return blocked ? 'blocked' : oldTarget ? 'upgrade' : 'install';
}
function shouldInstallHarness() {
    var state = harnessState();
    return state === 'install' || state === 'upgrade';
}
function startupVolume() {
    if (my.target.mountpoint === '/') return true;
    my.result.type = 'Fatal';
    my.result.message = system.localizedString('startup.volume');
    return false;
}
