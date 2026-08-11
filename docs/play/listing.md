# Google Play listing — TTS Runner

Everything the Play Console asks for, kept next to the app so a submission is
copy-paste. Assets in this folder: `icon-512.png` (hi-res icon),
`feature-graphic-1024x500.png`, and the phone screenshots in
`../screenshots/`. Privacy policy: `PRIVACY.md` (host it at a public URL and
paste that URL into the console).

## App details

- **App name:** TTS Runner
- **Package (applicationId):** `com.techhurts.ttsrunner`
- **Category:** Tools (alt: Productivity)
- **Default language:** English (United States)
- **Contact email:** weepingangles@gmail.com

## Short description (≤ 80 chars)

> On-device text-to-speech and voice cloning. No server, no account, no cloud.

## Full description (≤ 4000 chars)

> TTS Runner reads text and articles aloud in a voice you choose — entirely on
> your phone. No account, no server, nothing leaves the device.
>
> Share any web page or selected text and it strips the navigation, shows you
> the cleaned-up article to edit, then reads it aloud. A live player draws the
> waveform as it speaks and highlights the line being read; pause, save, or
> share the audio.
>
> Two engines ship side by side:
> • Supertonic 3 — faster than real time, 31 languages, style voices.
> • Qwen3-TTS — clones a voice from a short reference recording you make.
>
> Features:
> • Speakers — record a voice, design one from a description, or clone from an
>   audio file (trim it on a waveform first). Refine a cloned voice on the phone.
> • Chats — build a conversation with multiple speakers, reorder lines, replay
>   the whole thing or export it as one track. Set the pause between speakers.
> • Jobs — read to the screen or save to an m4a file in the background, with
>   resumable generation that survives the OS killing the job.
> • Hosting — optionally serve your voices as an OpenAI-compatible speech API to
>   other devices on your own network.
>
> Everything runs locally. The microphone is used only when you record a
> reference voice, and those recordings stay in the app's private storage. The
> network is used only for actions you start: downloading models, fetching a web
> page you share, or the optional local server.

## Screenshots (phone)

`../screenshots/speakers.png`, `player.png`, `jobs.png`, `chats.png`,
`hosting.png`, `settings.png` — 1080×2340, Galaxy S24 FE. Play needs 2–8;
these six are ready.

## Content rating questionnaire

- No violence, sexual, or profane content. User-generated audio is created
  locally by the user; no sharing platform inside the app. Expected rating:
  Everyone.

## Data safety form

- **Data collected / shared: none.** No personal or device data is collected or
  transmitted to the developer.
- **Data processed on-device only:** audio (microphone recordings), text the
  user provides. Not sent off device.
- **Security:** data stays in app-private storage; the user can delete it in-app.
- Answer "No" to data collection and data sharing. Declare that the app does not
  collect data. (The microphone is a permission, not data collection — no audio
  leaves the device.)

## Permissions justification (for the console, if prompted)

- `RECORD_AUDIO` — record a reference voice to create/clone a speaker; on-device
  only.
- `INTERNET` — download models; fetch a shared web page to read; optional local
  hosting server.
- `FOREGROUND_SERVICE*`, `WAKE_LOCK`, `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` —
  keep long/background speech generation alive (Samsung throttles otherwise).
- `POST_NOTIFICATIONS` — progress notification for a running/saved job.

## Pre-launch checklist

- [ ] Host `PRIVACY.md` at a public URL; paste it into the console.
- [ ] Upload `icon-512.png` and `feature-graphic-1024x500.png`.
- [ ] Upload the six phone screenshots.
- [ ] Enroll in Play App Signing (upload key = the repo's release keystore).
- [ ] Confirm the release AAB targets API 35 and passes the 16 KB page-size and
      pre-launch reports (see `../16kb-and-store-notes.md`).
