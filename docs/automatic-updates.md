# Automatic updates

[简体中文](zh-CN/automatic-updates.md)

## Check and install an update

In an update-enabled RC or stable App, choose **Check for Updates...** from the menu bar menu.
Follow the updater's prompts to download and install an available version. Automatic checks and downloads follow the updater's settings.
Save work in connected clients before restarting; quitting the App stops the backend and interrupts active work.

If checking fails, the installed App remains available. Check the network and try again later.
Ordinary local source builds have updates disabled by default; an unavailable update action does not mean a newer version exists.

## RC and stable channels

RC and stable releases use separate update channels. An RC installation does not automatically switch to the stable channel.
To move from RC to stable, deliberately install the stable App when it is available. Check the release notes before changing versions.
The updater compares globally increasing build numbers. RC suffixes such as `rc.1` are candidate sequence labels and do not reset or define the App build number.

## Downloads and privacy

Update checks contact the project's GitHub-hosted feed. Update downloads use the release asset URL in that feed.
The updater verifies update signatures before installation. These network requests do not upload local models or inference conversations.
See [Privacy](privacy.md) for network activity and [Data locations](storage-and-uninstall.md) before manually removing or replacing data.

Signing keys, feed publication and failure recovery are documented in the [release pipeline](stable-release-pipeline.md).
