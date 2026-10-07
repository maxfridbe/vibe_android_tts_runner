// Run after kitten-controls.ts with Kitten installed. Downloads Supertonic if needed.
device.start();
device.install(files.latest("output/tts-runner/tts-runner-*-debug.apk"));
device.apps.stop("com.techhurts.ttsrunner");
device.launch("com.techhurts.ttsrunner/.MainActivity");
device.getButton({ desc: "Chats" }).click();
device.getButton({ contains: "New chat" }).click();
expect(device.getButton({ desc: "Vocal event" })).toBeVisible();
log(device.shot("chat-delivery-controls"));
device.keys.back();
device.shell("am start -a android.intent.action.SEND -t text/plain --es android.intent.extra.TEXT 'Hello from the shared text editor.' -n com.techhurts.ttsrunner/.ShareActivity");
expect(device.getButton({ desc: "Emotion" })).toBeVisible();
expect(device.getButton({ desc: "Emphasize" })).toBeVisible();
log(device.shot("kitten-share-controls"));
device.keys.back();
device.getButton({ desc: "Settings" }).click();
device.getButton({ desc: "Model supertonic-3" }).scrollIntoView();
device.getButton({ desc: "Model supertonic-3" }).click();
device.getButton({ desc: "Download selected model" }).scrollIntoView();
device.getButton({ desc: "Download selected model" }).click();
expect(device.getLabel({ desc: "Model download status" })).toHaveText("Downloaded", { timeout: 1800000 });
device.shell("sync");
device.getButton({ desc: "Speakers" }).click();
device.getButton({ desc: "Jobs" }).click();
device.getButton({ desc: "Choose speaker" }).click();
device.getButton({ contains: "M1" }).click();
const field = device.getField({ desc: "Text to read" });
field.scrollIntoView(); field.click();
device.shell("input keycombination 113 29"); device.keys.press("67");
field.append("I see. Well then.");
device.keys.back();
device.getButton({ desc: "Vocal event" }).scrollIntoView();
expect(device.getButton({ desc: "Emotion" })).not.toBeVisible();
expect(device.getButton({ desc: "Emphasize" })).not.toBeVisible();
device.getButton({ desc: "Vocal event" }).click();
device.getButton({ text: "<breath>" }).click();
expect(field).toContainText("<breath>");
log(device.shot("supertonic-delivery-controls"));
device.getButton({ text: "Save file" }).click();
const started = Date.now();
device.getButton({ contains: "Add job" }).scrollIntoView();
device.getButton({ contains: "Add job" }).click();
let job;
const deadline = Date.now() + 300000;
while (Date.now() < deadline) {
  job = JSON.parse(device.shell("run-as com.techhurts.ttsrunner cat files/jobs.json"))
    .find((j) => j.id >= started - 10000 && j.text.includes("<breath>"));
  if (job && job.status !== "running") break;
  device.sleep(2000);
}
if (!job || job.status !== "done" || job.model !== "supertonic-3" || job.audioSecs <= 0) {
  throw new Error("Supertonic expression synthesis failed: " + JSON.stringify(job));
}
log(JSON.stringify(job));
device.getButton({ desc: "Chats" }).click();
device.getButton({ contains: "New chat" }).click();
expect(device.getButton({ desc: "Vocal event" })).toBeVisible();
expect(device.getButton({ desc: "Emotion" })).not.toBeVisible();
log(device.shot("supertonic-chat-controls"));
device.keys.back();
device.shell("am start -a android.intent.action.SEND -t text/plain --es android.intent.extra.TEXT 'Hello from the shared text editor.' -n com.techhurts.ttsrunner/.ShareActivity");
device.getButton({ contains: "M1" }).click();
expect(device.getButton({ desc: "Vocal event" })).toBeVisible();
expect(device.getButton({ desc: "Emotion" })).not.toBeVisible();
log(device.shot("supertonic-share-controls"));
