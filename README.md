# speech-detector

NodeJS library providing VAD (voice activity detection) or more specifically speech
activity detection. This library processes a raw stream of PCM audio data and
emits a stream of PCM audio data segements that contain speech. This library leverages
the `Silero` model for speech detection along with the ONNX framework.

## Install

```bash
npm install --save "speech-detector"
```

## Usage

```ts
import { SpeechDetector } from "speech-detector";

// Create a SpeechDetector using all default values.
const speechDetector = await SpeechDetector.create();

const speechSegments = await speechDetector.process(audioStream);

for await (const segement of speechSegments) {
  console.log(`Received speech segement: ${segement}`);
}
```
