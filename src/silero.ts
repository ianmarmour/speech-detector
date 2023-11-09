import ort from "onnxruntime-web";
import * as fs from "fs";

// This resolves a bug with WASM in nodejs.
ort.env.wasm.numThreads = 1;
ort.env.remoteModels = false;

class Silero {
  private session: ort.InferenceSession;
  private sampleRate: ort.Tensor;

  // Internal variables
  private c: ort.Tensor;
  private h: ort.Tensor;

  private constructor(session: ort.InferenceSession, sampleRate: number) {
    this.session = session;
    // Note: I suppose the model requires a tensor as input.
    this.sampleRate = new ort.Tensor("int64", [sampleRate]);

    // Note: No idea what these inputs to the model are for...
    this.c = new ort.Tensor("float32", Array(2 * 64).fill(0), [2, 1, 64]);
    this.h = new ort.Tensor("float32", Array(2 * 64).fill(0), [2, 1, 64]);
  }

  static async create(
    sampleRate: number,
    uri: string = "./model/silero_vad.onnx"
  ) {
    const opt: ort.InferenceSession.SessionOptions = {
      executionProviders: ["wasm"],
      logSeverityLevel: 3,
      logVerbosityLevel: 3,
    };

    // For compatability convert the URI into a properly
    // formatted URL. This will work for NodeJS and Web.
    const path = new URL(uri, import.meta.url);

    let session: ort.InferenceSession;

    if (typeof window === "undefined") {
      // Only read in the model file in NodeJS.
      const model = fs.readFileSync(path);

      session = await ort.InferenceSession.create(model, opt);
    } else {
      session = await ort.InferenceSession.create(uri, opt);
    }

    return new Silero(session, sampleRate);
  }

  async process(audio: Float32Array) {
    const t = new ort.Tensor("float32", audio, [1, audio.length]);

    const input = {
      input: t,
      h: this.h,
      c: this.c,
      sr: this.sampleRate,
    };

    const output = await this.session.run(input);

    this.h = output.hn;
    this.c = output.cn;

    return output.output.data[0] as number;
  }
}

export { Silero };
