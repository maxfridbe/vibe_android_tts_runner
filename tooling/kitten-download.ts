// Run from the repository root with android-screen-runner.
device.start();
device.install(files.latest("output/tts-runner/tts-runner-*-debug.apk"));
device.apps.stop("com.techhurts.ttsrunner");
device.launch("com.techhurts.ttsrunner/.MainActivity", {
  grant: ["android.permission.POST_NOTIFICATIONS"],
});
device.getButton({ desc: "Settings" }).click();
device.getButton({ desc: "Model kitten-tts-2" }).scrollIntoView();
device.getButton({ desc: "Model kitten-tts-2" }).click();
device.getButton({ desc: "Download selected model" }).scrollIntoView();
device.getButton({ desc: "Download selected model" }).click();
expect(device.getLabel({ desc: "Model download status" })).toHaveText("Downloaded", { timeout: 1800000 });
device.shell("sync");
log(device.shot("kitten-model-downloaded"));
