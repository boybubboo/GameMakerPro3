export interface CompressedMakerResult {
  blob: Blob;
  filename: string;
  rawBytes: number;
  compressedBytes: number;
  compressed: boolean;
}

export function projectToMakerFilename(name: string): string {
  const slug = name
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return `${slug || 'maker-game'}.maker`;
}

/**
 * Compresses a serialized Maker JSON bundle into a compact gzip-compressed `.maker` binary Blob.
 * Uses native Web Streams CompressionStream('gzip') with a safe uncompressed fallback if unavailable.
 */
export async function compressMakerBundle(
  jsonText: string,
  projectName: string
): Promise<CompressedMakerResult> {
  const encoder = new TextEncoder();
  const rawUint8 = encoder.encode(jsonText);
  const rawBytes = rawUint8.byteLength;
  const filename = projectToMakerFilename(projectName);

  if (typeof CompressionStream !== 'undefined') {
    try {
      const compressedStream = new Blob([rawUint8])
        .stream()
        .pipeThrough(new CompressionStream('gzip'));
      const buffer = await new Response(compressedStream).arrayBuffer();
      return {
        blob: new Blob([buffer], { type: 'application/octet-stream' }),
        filename,
        rawBytes,
        compressedBytes: buffer.byteLength,
        compressed: true,
      };
    } catch {
      // Fall back to plain UTF-8 bytes if gzip stream fails
    }
  }

  return {
    blob: new Blob([rawUint8], { type: 'application/json' }),
    filename,
    rawBytes,
    compressedBytes: rawBytes,
    compressed: false,
  };
}

/**
 * Triggers a browser download for a Blob using the given filename.
 */
export function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/**
 * Reads either a gzip-compressed binary `.maker` file or a plain UTF-8 `.maker` / `.json` file
 * and returns the decoded JSON string.
 */
export async function readMakerFileAsText(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  // Detect gzip magic bytes (0x1F, 0x8B)
  const isGzip = bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
  if (isGzip) {
    if (typeof DecompressionStream === 'undefined') {
      throw new Error('This browser does not support decompressing compressed .maker files.');
    }
    const decompressedStream = new Blob([bytes])
      .stream()
      .pipeThrough(new DecompressionStream('gzip'));
    return await new Response(decompressedStream).text();
  }

  return new TextDecoder('utf-8').decode(bytes);
}
