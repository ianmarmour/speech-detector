import { ReadableStream } from "node:stream/web";
import { Silero } from "./silero.js";

type SpeechSegment = {
  startSampleIndex: number;
  endSampleIndex: number;
  audioSegment: Float32Array;
};

class SpeechDetector {
  private silero: Silero;
  private frameSamples: number;
  private positiveSpeechThreshold: number;
  private negativeSpeechThreshold: number;
  private redemptionFrames: number;
  private minSpeechFrames: number;
  private speechSegmentsStreamController: ReadableStreamDefaultController<Float32Array> | null =
    null;
  private currentSpeechSegment: SpeechSegment | null = null;
  private speaking = false;
  private redemptionCounter = 0;
  private speechFrameCount = 0;
  private frameCount = 0;

  private constructor(
    silero: Silero,
    frameSamples: number,
    positiveSpeechThreshold: number,
    negativeSpeechThreshold: number,
    redemptionFrames: number,
    minSpeechFrames: number
  ) {
    this.silero = silero;
    this.frameSamples = frameSamples;
    this.positiveSpeechThreshold = positiveSpeechThreshold;
    this.negativeSpeechThreshold = negativeSpeechThreshold;
    this.redemptionFrames = redemptionFrames;
    this.minSpeechFrames = minSpeechFrames;
  }

  public static async create(
    frameSamples: number = 1536,
    positiveSpeechThreshold: number = 0.5,
    negativeSpeechThreshold: number = 0.5 - 0.15,
    redemptionFrames: number = 15,
    minSpeechFrames: number = 1
  ): Promise<SpeechDetector> {
    // Create an instance of the Silero VAD
    // model this only supports 16000hz PCM
    // audio at the moment.
    const silero = await Silero.create(16000);
    return new SpeechDetector(
      silero,
      frameSamples,
      positiveSpeechThreshold,
      negativeSpeechThreshold,
      redemptionFrames,
      minSpeechFrames
    );
  }

  private resetSpeechDetection(): void {
    this.currentSpeechSegment = null;
    this.speaking = false;
    this.redemptionCounter = 0;
    this.speechFrameCount = 0;
    this.frameCount = 0;
  }

  private emitSpeechSegment(segment: SpeechSegment): void {
    this.speechSegmentsStreamController?.enqueue(segment.audioSegment);
  }

  private startNewSpeechSegment(frame: Float32Array): void {
    this.speaking = true;
    this.speechFrameCount = 1;
    this.currentSpeechSegment = {
      startSampleIndex: this.frameCount * this.frameSamples,
      endSampleIndex: 0,
      audioSegment: frame.slice(),
    };
  }

  private appendToCurrentSpeechSegment(frame: Float32Array): void {
    if (this.currentSpeechSegment) {
      this.speechFrameCount++;
      const newAudioSegment = new Float32Array(
        this.currentSpeechSegment.audioSegment.length + frame.length
      );
      newAudioSegment.set(this.currentSpeechSegment.audioSegment);
      newAudioSegment.set(frame, this.currentSpeechSegment.audioSegment.length);
      this.currentSpeechSegment.audioSegment = newAudioSegment;
    }
  }

  private finalizeSpeechSegment(): void {
    if (
      this.currentSpeechSegment &&
      this.speechFrameCount >= this.minSpeechFrames
    ) {
      this.currentSpeechSegment.endSampleIndex =
        this.frameCount * this.frameSamples;
      this.emitSpeechSegment(this.currentSpeechSegment);
      this.resetSpeechDetection();
    }
  }

  private handleNonSpeechFrame(probability: number): void {
    if (probability < this.negativeSpeechThreshold) {
      this.redemptionCounter++;
      if (this.redemptionCounter >= this.redemptionFrames) {
        this.finalizeSpeechSegment();
        this.speaking = false;
        this.speechFrameCount = 0;
        this.redemptionCounter = 0;
      }
    }
  }

  private handleSpeechProbability(
    probability: number,
    frame: Float32Array
  ): void {
    if (probability > this.positiveSpeechThreshold) {
      if (!this.speaking) {
        this.startNewSpeechSegment(frame);
      } else {
        this.appendToCurrentSpeechSegment(frame);
      }
      this.redemptionCounter = 0;
    } else if (this.speaking) {
      this.handleNonSpeechFrame(probability);
    }

    this.frameCount++;
  }

  public process(
    audio: ReadableStream<Float32Array>
  ): ReadableStream<Float32Array> {
    const speechSegmentsStream = new ReadableStream<Float32Array>({
      start: (controller) => {
        this.speechSegmentsStreamController = controller;
      },
      cancel: () => {
        this.resetSpeechDetection();
      },
    });

    const reader = audio.getReader();

    const processFrame = async (frame: Float32Array) => {
      const probability = await this.silero.process(frame);
      this.handleSpeechProbability(probability, frame);
    };

    const readAndProcess = () => {
      reader
        .read()
        .then(({ done, value }) => {
          if (done) {
            this.speechSegmentsStreamController?.close();
            return;
          }
          if (value) {
            this.processAudioChunk(value, processFrame, readAndProcess);
          }
        })
        .catch((error) => {
          this.speechSegmentsStreamController?.error(error);
        });
    };

    readAndProcess();

    return speechSegmentsStream;
  }

  private async processAudioChunk(
    audioChunk: Float32Array,
    processFrame: (frame: Float32Array) => Promise<void>,
    readAndProcess: () => void
  ): Promise<void> {
    for (let i = 0; i < audioChunk.length; i += this.frameSamples) {
      const frame = audioChunk.slice(i, i + this.frameSamples);
      await processFrame(frame);
    }
    readAndProcess();
  }
}

export { SpeechDetector };
