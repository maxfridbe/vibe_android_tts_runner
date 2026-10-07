// Run after kitten-smoke.ts: verify cancellation and retain resumable metadata.
device.start();
device.install(files.latest("output/tts-runner/tts-runner-*-debug.apk"));
device.apps.stop("com.techhurts.ttsrunner");
device.launch("com.techhurts.ttsrunner/.MainActivity");
device.getButton({ desc: "Speakers" }).click();
device.getButton({ desc: "Kitten speakers" }).click();
device.getButton({ desc: "Use Kitten: Bruno" }).scrollIntoView();
device.getButton({ desc: "Use Kitten: Bruno" }).click();
const sentence = "This is the Kitten cancellation test. ".repeat(12).trim();
const field = device.getField({ desc: "Text to read" });
field.click();
device.shell("input keycombination 113 29");
device.keys.press("67");
field.append(sentence);
expect(field).toHaveText(sentence);
device.keys.back();
device.getButton({ text: "Save file" }).click();
const started = Date.now();
device.getButton({ contains: "Add job" }).scrollIntoView();
device.getButton({ contains: "Add job" }).click();
device.getButton({ contains: "Stop" }).click();
let job;
const deadline = Date.now() + 120000;
// Job status is the observable completion signal; native cancellation is asynchronous.
while (Date.now() < deadline) {
  const jobs = JSON.parse(device.shell("run-as com.techhurts.ttsrunner cat files/jobs.json"));
  job = jobs.find((entry) => entry.id >= started - 10000 && entry.model === "kitten-tts-2" && entry.text === sentence);
  if (job && job.status !== "running") break;
  device.sleep(2000);
}
if (!job || job.status !== "stopped" || job.chunksTotal <= 1) {
  throw new Error("Kitten cancellation failed: " + JSON.stringify(job));
}
log(JSON.stringify(job));
log(device.shot("kitten-canceled"));

// Start again without force-stopping the process: the native cancel flag must reset.
field.scrollIntoView(); field.click();
device.shell("input keycombination 113 29");
device.keys.press("67");
field.append("Hello again. Kitten can speak after cancellation.");
device.keys.back();
const recoveryStarted = Date.now();
device.getButton({ contains: "Add job" }).scrollIntoView();
device.getButton({ contains: "Add job" }).click();
const recoveryDeadline = Date.now() + 900000;
while (Date.now() < recoveryDeadline) {
  const jobs = JSON.parse(device.shell("run-as com.techhurts.ttsrunner cat files/jobs.json"));
  job = jobs.find((entry) => entry.id >= recoveryStarted - 10000 && entry.model === "kitten-tts-2" && entry.text === "Hello again. Kitten can speak after cancellation.");
  if (job && job.status !== "running") break;
  device.sleep(2000);
}
if (!job || job.status !== "done" || job.audioSecs <= 0) {
  throw new Error("Kitten did not recover after cancellation: " + JSON.stringify(job));
}
log(JSON.stringify(job));
log(device.shot("kitten-recovered"));
