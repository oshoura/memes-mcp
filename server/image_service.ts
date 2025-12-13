import fs from "fs";
import { Jimp, loadFont, measureTextHeight } from "jimp";
import {
  SANS_8_WHITE,
  SANS_16_WHITE,
  SANS_32_WHITE,
  SANS_64_WHITE,
  SANS_128_WHITE,
} from "jimp/fonts";
import {
  SANS_8_BLACK,
  SANS_16_BLACK,
  SANS_32_BLACK,
  SANS_64_BLACK,
  SANS_128_BLACK,
} from "jimp/fonts";

type JimpFont = any;
type JimpImage = Awaited<ReturnType<typeof Jimp.read>>;

const whiteFonts: any[] = [SANS_128_WHITE, SANS_64_WHITE, SANS_32_WHITE, SANS_16_WHITE, SANS_8_WHITE];
const blackFonts: any[] = [SANS_128_BLACK, SANS_64_BLACK, SANS_32_BLACK, SANS_16_BLACK, SANS_8_BLACK];
const loadedFontCache: Record<string, JimpFont> = {};

async function loadFontCached(fontRef: any): Promise<JimpFont> {
  const key = String(fontRef);
  if (!loadedFontCache[key]) {
    loadedFontCache[key] = await loadFont(fontRef);
  }
  return loadedFontCache[key];
}

async function pickFontPairForBox(
  text: string,
  boxWidth: number,
  boxHeight: number
): Promise<{ whiteFont: JimpFont; blackFont: JimpFont; strokePx: number }> {
  for (let i = 0; i < whiteFonts.length; i++) {
    const whiteRef = whiteFonts[i];
    const whiteFont = await loadFontCached(whiteRef);
    const heightNeeded = measureTextHeight(whiteFont, text, boxWidth);
    if (heightNeeded <= boxHeight) {
      const blackRef = blackFonts[i];
      const blackFont = await loadFontCached(blackRef);
      const strokePx = i === 0 ? 4 : i === 1 ? 3 : i === 2 ? 2 : 1;
      return { whiteFont, blackFont, strokePx };
    }
  }

  const lastIdx = whiteFonts.length - 1;
  return {
    whiteFont: await loadFontCached(whiteFonts[lastIdx]),
    blackFont: await loadFontCached(blackFonts[lastIdx]),
    strokePx: 1,
  };
}

export type TextBox = {
  text: string;
  left: number;
  top: number;
  width: number;
  height: number;
};

export type TextPosition = {
  left: number;
  top: number;
  width: number;
  height: number;
};

export type MemeTextOption = {
  position?: TextPosition;
  updated_position?: TextPosition;
};

export function buildTextBoxesFromOptions(
  texts: Array<{ id: string; text: string }>,
  options: MemeTextOption[]
): TextBox[] {
  const textMap = new Map<string, string>();
  for (const t of texts) textMap.set(t.id, t.text);

  const boxes: TextBox[] = [];
  for (let i = 0; i < options.length; i++) {
    const key = String(i);
    const value = textMap.get(key);
    if (!value) continue;

    const pos = options[i]?.updated_position || options[i]?.position;
    if (!pos) continue;

    const left = Math.floor(pos.left);
    const top = Math.floor(pos.top);
    const width = Math.max(1, Math.floor(pos.width));
    const height = Math.max(1, Math.floor(pos.height));

    boxes.push({ text: value, left, top, width, height });
  }

  return boxes;
}

export class Image {
  private constructor(private img: JimpImage) { }

  static async fromPath(imagePath: string): Promise<Image> {
    let buffer: Buffer;
    if (/^https?:\/\//i.test(imagePath)) {
      const response = await fetch(imagePath);
      const arr = await response.arrayBuffer();
      buffer = Buffer.from(arr);
    } else {
      buffer = fs.readFileSync(imagePath);
    }
    const img = await Jimp.read(buffer);
    return new Image(img);
  }

  static async fromBuffer(buffer: Buffer): Promise<Image> {
    const img = await Jimp.read(buffer);
    return new Image(img);
  }

  get width(): number {
    return this.img.width;
  }

  get height(): number {
    return this.img.height;
  }

  resizeTo(width?: number, height?: number): this {
    if (width && height && (this.img.width !== width || this.img.height !== height)) {
      this.img.resize({ w: width, h: height });
    }
    return this;
  }

  async addTexts(boxes: TextBox[]): Promise<void> {
    for (const box of boxes) {
      const { whiteFont, blackFont, strokePx } = await pickFontPairForBox(box.text, box.width, box.height);
      const offsets: Array<[number, number]> = [
        [-strokePx, 0],
        [strokePx, 0],
        [0, -strokePx],
        [0, strokePx],
        [-strokePx, -strokePx],
        [-strokePx, strokePx],
        [strokePx, -strokePx],
        [strokePx, strokePx],
      ];

      for (const [dx, dy] of offsets) {
        this.img.print({
          font: blackFont,
          x: box.left + dx,
          y: box.top + dy,
          text: box.text,
          maxWidth: box.width,
          maxHeight: box.height,
        });
      }

      this.img.print({
        font: whiteFont,
        x: box.left,
        y: box.top,
        text: box.text,
        maxWidth: box.width,
        maxHeight: box.height,
      });
    }
  }

  async save(outPath: string): Promise<void> {
    await this.img.write(outPath as `${string}.png`);
  }

  async toPngBuffer(): Promise<Buffer> {
    return await this.img.getBuffer("image/png");
  }
}

