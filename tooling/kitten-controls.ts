// Android x86_64 emulator: GPU absence must fail explicitly, then CPU must recover.
device.start();
device.install(files.latest("output/tts-runner/tts-runner-*-debug.apk"));
device.apps.stop("com.techhurts.ttsrunner");
device.launch("com.techhurts.ttsrunner/.MainActivity", { grant: ["android.permission.POST_NOTIFICATIONS"] });
device.getButton({ desc: "Settings" }).click();
device.getButton({ desc: "Backend kitten opencl" }).scrollIntoView();
log(device.shot("kitten-gpu-options"));
device.getButton({ desc: "Backend kitten opencl" }).click();
device.getButton({ desc: "Speakers" }).click();
device.getButton({ desc: "Kitten speakers" }).click();
device.getButton({ desc: "Use Kitten: Bruno" }).scrollIntoView();
device.getButton({ desc: "Use Kitten: Bruno" }).click();
const field = device.getField({ desc: "Text to read" });
field.click();
device.shell("input keycombination 113 29"); device.keys.press("67");
field.append("Testing the Kitten GPU choice.");
device.keys.back();
device.getButton({ text: "Save file" }).click();
let started = Date.now();
device.getButton({ contains: "Add job" }).scrollIntoView();
device.getButton({ contains: "Add job" }).click();
let job;
let deadline = Date.now() + 120000;
while (Date.now() < deadline) {
  job = JSON.parse(device.shell("run-as com.techhurts.ttsrunner cat files/jobs.json"))
    .find((j) => j.id >= started - 10000 && j.text === "Testing the Kitten GPU choice.");
  if (job && job.status !== "running") break;
  device.sleep(2000);
}
if (!job || job.backend !== "opencl" || !job.error.includes("No opencl device available")) {
  throw new Error("Expected explicit unavailable-GPU error: " + JSON.stringify(job));
}
log(JSON.stringify(job));
log(device.shot("kitten-gpu-unavailable"));
device.getButton({ desc: "Settings" }).click();
device.getButton({ desc: "Backend kitten cpu" }).scrollIntoView();
device.getButton({ desc: "Backend kitten cpu" }).click();
device.getButton({ desc: "Jobs" }).click();
field.scrollIntoView(); field.click();
device.shell("input keycombination 113 29"); device.keys.press("67");
field.append("Hello, my friend.");
device.keys.back();
device.getButton({ desc: "Emotion" }).scrollIntoView();
device.getButton({ desc: "Emotion" }).click();
device.getButton({ text: "joyful" }).click();
device.getButton({ desc: "Vocal event" }).click();
device.getButton({ text: "<laugh>" }).click();
device.getButton({ desc: "Emphasize" }).click();
device.getField({ desc: "Word or short phrase" }).fill("Welcome");
device.getButton({ text: "Insert" }).click();
expect(field).toContainText("[joyful] Hello, my friend.");
expect(field).toContainText("<laugh>");
expect(field).toContainText("(((Welcome)))");
log(device.shot("kitten-delivery-controls"));
started = Date.now();
device.getButton({ contains: "Add job" }).scrollIntoView();
device.getButton({ contains: "Add job" }).click();
deadline = Date.now() + 900000;
while (Date.now() < deadline) {
  job = JSON.parse(device.shell("run-as com.techhurts.ttsrunner cat files/jobs.json"))
    .find((j) => j.id >= started - 10000 && j.text.includes("(((Welcome)))"));
  if (job && job.status !== "running") break;
  device.sleep(2000);
}
if (!job || job.status !== "done" || job.backend !== "cpu" || job.audioSecs <= 0) {
  throw new Error("Kitten expression synthesis failed: " + JSON.stringify(job));
}
log(JSON.stringify(job));
log(device.shot("kitten-expression-complete"));
