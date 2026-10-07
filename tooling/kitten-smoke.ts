// Requires the Kitten model downloaded by kitten-download.ts.
device.start();
device.apps.stop("com.techhurts.ttsrunner");
device.launch("com.techhurts.ttsrunner/.MainActivity", {
  grant: ["android.permission.POST_NOTIFICATIONS"],
});
device.getButton({ desc: "Speakers" }).click();
device.getButton({ desc: "Kitten speakers" }).click();
expect(device.getLabel({ contains: "Kitten TTS 2 speakers" })).toBeVisible();
log(device.shot("kitten-speakers"));
device.getButton({ desc: "Use Kitten: Bruno" }).scrollIntoView();
device.getButton({ desc: "Use Kitten: Bruno" }).click();
const sentence = "Hello. This is Kitten speaking on Android.";
const field = device.getField({ desc: "Text to read" });
field.click();
// Ctrl+A selects the entire multiline editor; MOVE_END only reaches a line end.
device.shell("input keycombination 113 29");
device.keys.press("67");
field.append(sentence);
expect(field).toHaveText(sentence);
device.keys.back();
device.getButton({ text: "Save file" }).click();
const started = Date.now();
device.getButton({ contains: "Add job" }).click();
// The runner has no job-state wait. Poll the persisted job instead of dumping
// UI hierarchies while all emulator cores are occupied by native inference.
let job;
const deadline = Date.now() + 900000;
while (Date.now() < deadline) {
  const jobs = JSON.parse(device.shell("run-as com.techhurts.ttsrunner cat files/jobs.json"));
  job = jobs.find((entry) => entry.model === "kitten-tts-2" && entry.text === sentence && entry.id >= started - 10000);
  if (job && job.status !== "running") break;
  device.sleep(2000);
}
log(device.shot("kitten-job-complete"));
if (!job || job.status !== "done" || job.audioSecs <= 0 || job.voice !== "Kitten: Bruno") {
  throw new Error("Kitten did not finish a non-empty synthesis job: " + JSON.stringify(job));
}
log(JSON.stringify(job));
device.shell("run-as com.techhurts.ttsrunner cat files/last_audio.wav > /data/local/tmp/kitten-smoke.wav");
device.files.pull("/data/local/tmp/kitten-smoke.wav", "screenshots/kitten-smoke.wav");
device.files.remove("/data/local/tmp/kitten-smoke.wav");
