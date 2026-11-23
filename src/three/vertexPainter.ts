import * as THREE from 'three';
import {
  attribute,
  materialColor,
  materialOpacity,
  vec4,
  vertexColor,
} from 'three/tsl';
import { applyHsvShiftLinearTsl, upgradeToNodeMaterial } from './nodeMaterialUtils';
import type {
  VertexPaintMode,
  VertexPaintStroke,
  VertexPaintTarget,
} from '../types/maker';

export interface VertexPaintBrushSettings {
  radius: number;
  opacity: number;
  softness: number;
  mode: VertexPaintMode;
  target: VertexPaintTarget;
  color: string;
  hueShift: number; // -180..+180 degrees
  saturationShift: number; // -0.8..+0.8
  valueShift: number; // -0.5..+0.5
  hueJitter: number; // 0..90 degrees
}

// Reusable math scratch objects (zero GC during live pointer-move brushing)
const _invRootMat = new THREE.Matrix4();
const _meshToRootMat = new THREE.Matrix4();
const _vPos = new THREE.Vector3();
const _srgbA = { r: 0, g: 0, b: 0 };
const _srgbB = { r: 0, g: 0, b: 0 };
const _cTmp = new THREE.Color();
const _cCurr = new THREE.Color();
const _cTarget = new THREE.Color();
const _cBase = new THREE.Color();
const _hslTmp = { h: 0, s: 0, l: 0 };

function hashVertex3(x: number, y: number, z: number, seed: number): number {
  const xi = Math.round(x * 1000);
  const yi = Math.round(y * 1000);
  const zi = Math.round(z * 1000);
  let h = (seed | 0) ^ Math.imul(xi, 374761393) ^ Math.imul(yi, 668265263) ^ Math.imul(zi, 2147483647);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/**
 * Smooth spatial 3D wave noise in [-1..+1] for organic painterly foliage/bark mottling.
 */
function organicSpatialNoise(x: number, y: number, z: number, seed: number): number {
  const s = (seed % 997) * 0.013;
  const n1 = Math.sin(x * 3.1 + s) * Math.cos(y * 3.7 - s * 0.7) * Math.sin(z * 2.9 + s * 1.1);
  const n2 = Math.sin(x * 7.3 - z * 6.1 + s * 2.3) * Math.cos(y * 6.7 + x * 4.2);
  return THREE.MathUtils.clamp(n1 * 0.65 + n2 * 0.35, -1, 1);
}

/**
 * Shifts a hex color in sRGB HSL space by hue degrees, saturation delta, and lightness delta.
 */
export function computeShiftedHexColor(
  hex: string,
  hueShiftDeg: number,
  satShift = 0,
  valShift = 0
): string {
  const c = new THREE.Color(hex);
  c.getHSL(_hslTmp, THREE.SRGBColorSpace);
  const nh = (((_hslTmp.h + hueShiftDeg / 360) % 1) + 1) % 1;
  const ns = THREE.MathUtils.clamp(_hslTmp.s + satShift, 0, 1);
  const nl = THREE.MathUtils.clamp(_hslTmp.l + valShift * 0.55, 0.04, 0.96);
  c.setHSL(nh, ns, nl, THREE.SRGBColorSpace);
  return `#${c.getHexString()}`;
}

/**
 * Ensures a BufferGeometry has mutable `color` (RGB linear) and `aPaintHsv` (hue, sat, val)
 * vertex attributes plus immutable baseline snapshots for non-destructive erasing/resetting.
 */
export function ensureGeometryVertexPaintAttributes(geo: THREE.BufferGeometry): void {
  const pos = geo.attributes.position as THREE.BufferAttribute | undefined;
  if (!pos) return;
  const count = pos.count;

  let colAttr = geo.attributes.color as THREE.BufferAttribute | undefined;
  if (!colAttr || colAttr.count !== count || colAttr.itemSize !== 3) {
    const arr = new Float32Array(count * 3);
    arr.fill(1.0);
    colAttr = new THREE.BufferAttribute(arr, 3);
    geo.setAttribute('color', colAttr);
  } else if (!(colAttr.array instanceof Float32Array)) {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      arr[i * 3] = colAttr.getX(i);
      arr[i * 3 + 1] = colAttr.getY(i);
      arr[i * 3 + 2] = colAttr.getZ(i);
    }
    colAttr = new THREE.BufferAttribute(arr, 3);
    geo.setAttribute('color', colAttr);
  }

  let hsvAttr = geo.attributes.aPaintHsv as THREE.BufferAttribute | undefined;
  if (!hsvAttr || hsvAttr.count !== count || hsvAttr.itemSize !== 3) {
    hsvAttr = new THREE.BufferAttribute(new Float32Array(count * 3), 3);
    geo.setAttribute('aPaintHsv', hsvAttr);
  }

  if (!geo.userData.vpBaseColors) {
    geo.userData.vpBaseColors = new Float32Array(colAttr.array as Float32Array);
  }
  if (!geo.userData.vpBaseHsv) {
    geo.userData.vpBaseHsv = new Float32Array(hsvAttr.array as Float32Array);
  }
}

/**
 * Upgrades/patches any standard/toon/physical/basic material with per-vertex `aPaintHsv`
 * TSL colorNode RGB<->HSV hue/saturation/value shifting so Hue Shift vertex painting
 * preserves 100% of underlying bark, leaf, or uploaded seamless textures under WebGPURenderer!
 */
export function enableVertexPaintHsvOnMaterial(mat: THREE.Material): THREE.Material {
  if (!mat) return mat;

  // Materials already patched by patchCanopyMaterial with enableVertexPaintHsv already support aPaintHsv
  if (mat.userData.hasVertexPaintHsv) {
    if ('vertexColors' in mat) {
      const std = mat as THREE.MeshStandardMaterial;
      if (!std.vertexColors && !mat.userData._vpBakedVertexColors) {
        std.vertexColors = true;
        std.needsUpdate = true;
      }
    }
    return mat;
  }

  const nodeMat = upgradeToNodeMaterial(mat);
  nodeMat.userData.hasVertexPaintHsv = true;
  nodeMat.userData._vpBakedVertexColors = true;
  // Fold vertex color inside colorNode before HSV shift so diffuse * vertexColor is HSV-shifted once
  nodeMat.vertexColors = false;

  const vPaintHsv = attribute('aPaintHsv', 'vec3');
  const combinedRgb = materialColor.rgb.mul(vertexColor().rgb);
  const shiftedRgb = applyHsvShiftLinearTsl(combinedRgb, vPaintHsv);
  nodeMat.colorNode = vec4(shiftedRgb.rgb, materialOpacity);
  nodeMat.needsUpdate = true;
  return nodeMat;
}

/**
 * Prepares all paintable meshes inside a 3D object/tree for interactive vertex painting:
 * ensures unique non-shared BufferGeometries, initializes `color` + `aPaintHsv` attributes,
 * snapshots original vertex colors for non-destructive erasing, and upgrades material shaders.
 */
export function prepareObjectForVertexPainting(root: THREE.Object3D): THREE.Mesh[] {
  const seenGeos = new Set<THREE.BufferGeometry>();
  const paintableMeshes: THREE.Mesh[] = [];

  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    if (child.userData.noVertexPaint) return;

    // Clone geometry if shared between sibling meshes (e.g. multiple canopy puffs) so painting stays localized
    if (seenGeos.has(child.geometry)) {
      child.geometry = child.geometry.clone();
    }
    seenGeos.add(child.geometry);

    ensureGeometryVertexPaintAttributes(child.geometry);

    if (Array.isArray(child.material)) {
      child.material = child.material.map((m) => enableVertexPaintHsvOnMaterial(m));
    } else if (child.material) {
      child.material = enableVertexPaintHsvOnMaterial(child.material);
    }

    paintableMeshes.push(child);
  });

  return paintableMeshes;
}

/**
 * Restores every vertex on every paintable mesh in `root` back to its pristine unpainted baseline.
 */
export function resetObjectVertexPaint(root: THREE.Object3D): void {
  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh) || child.userData.noVertexPaint) return;
    const geo = child.geometry;
    const baseCols = geo.userData.vpBaseColors as Float32Array | undefined;
    const baseHsv = geo.userData.vpBaseHsv as Float32Array | undefined;
    const colAttr = geo.attributes.color as THREE.BufferAttribute | undefined;
    const hsvAttr = geo.attributes.aPaintHsv as THREE.BufferAttribute | undefined;
    if (baseCols && colAttr && colAttr.array instanceof Float32Array) {
      colAttr.array.set(baseCols);
      colAttr.needsUpdate = true;
    }
    if (baseHsv && hsvAttr && hsvAttr.array instanceof Float32Array) {
      hsvAttr.array.set(baseHsv);
      hsvAttr.needsUpdate = true;
    }
  });
}

function matchesTarget(mesh: THREE.Mesh, target: VertexPaintTarget): boolean {
  if (target === 'all') return true;
  const part = mesh.userData.paintPart as string | undefined;
  if (!part) return true;
  return part === target;
}

/**
 * Applies a single 3D spherical brush dab (in `root` local coordinates) directly to vertex
 * `color` and `aPaintHsv` buffers with smooth radial falloff. Runs in <1ms for 60fps brushing.
 */
export function applyVertexPaintDab(
  root: THREE.Object3D,
  localCenter: THREE.Vector3,
  settings: VertexPaintBrushSettings,
  seed: number,
  cachedMeshes?: THREE.Mesh[]
): boolean {
  root.updateMatrixWorld(true);
  _invRootMat.copy(root.matrixWorld).invert();

  const radius = Math.max(0.05, settings.radius);
  const radiusSq = radius * radius;
  const opacity = THREE.MathUtils.clamp(settings.opacity, 0.01, 1.0);
  const softness = THREE.MathUtils.clamp(settings.softness, 0.05, 1.0);
  const hueNorm = settings.hueShift / 360;
  const satShift = THREE.MathUtils.clamp(settings.saturationShift ?? 0, -0.8, 0.8);
  const valShift = THREE.MathUtils.clamp(settings.valueShift ?? 0, -0.5, 0.5);
  const jitterDeg = THREE.MathUtils.clamp(settings.hueJitter ?? 0, 0, 90);

  // Precompute base brush color shifted by hueShift/sat/val for 'paint' mode
  _cBase.set(settings.color);
  _cBase.getHSL(_hslTmp, THREE.SRGBColorSpace);
  const baseH = (((_hslTmp.h + hueNorm) % 1) + 1) % 1;
  const baseS = THREE.MathUtils.clamp(_hslTmp.s + satShift, 0, 1);
  const baseL = THREE.MathUtils.clamp(_hslTmp.l + valShift * 0.5, 0.04, 0.96);

  const meshes = cachedMeshes ?? [];
  if (!cachedMeshes) {
    root.traverse((child) => {
      if (child instanceof THREE.Mesh && !child.userData.noVertexPaint) {
        meshes.push(child);
      }
    });
  }

  let anyModified = false;

  for (let m = 0; m < meshes.length; m++) {
    const mesh = meshes[m];
    if (!mesh.visible) continue;
    if (!matchesTarget(mesh, settings.target)) continue;

    const geo = mesh.geometry;
    const posAttr = geo.attributes.position as THREE.BufferAttribute | undefined;
    const colAttr = geo.attributes.color as THREE.BufferAttribute | undefined;
    const hsvAttr = geo.attributes.aPaintHsv as THREE.BufferAttribute | undefined;
    if (!posAttr || !colAttr || !hsvAttr) continue;

    // Matrix transforming vertex position from mesh local space into `root` local space
    _meshToRootMat.multiplyMatrices(_invRootMat, mesh.matrixWorld);

    const baseCols = geo.userData.vpBaseColors as Float32Array | undefined;
    let meshTouched = false;

    for (let i = 0; i < posAttr.count; i++) {
      _vPos.set(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i)).applyMatrix4(_meshToRootMat);

      const dx = _vPos.x - localCenter.x;
      const dy = _vPos.y - localCenter.y;
      const dz = _vPos.z - localCenter.z;
      const distSq = dx * dx + dy * dy + dz * dz;
      if (distSq >= radiusSq) continue;

      const normDist = Math.sqrt(distSq) / radius;
      const innerCore = 1 - softness;
      let falloff = 1.0;
      if (normDist > innerCore) {
        const edgeT = (normDist - innerCore) / Math.max(0.001, softness);
        const smooth = edgeT * edgeT * (3 - 2 * edgeT);
        falloff = 1 - smooth;
      }

      const w = THREE.MathUtils.clamp(falloff * opacity, 0, 1);
      if (w <= 0.0005) continue;

      meshTouched = true;

      if (settings.mode === 'erase') {
        const i3 = i * 3;
        const br = baseCols ? baseCols[i3] : 1;
        const bg = baseCols ? baseCols[i3 + 1] : 1;
        const bb = baseCols ? baseCols[i3 + 2] : 1;
        colAttr.setXYZ(
          i,
          THREE.MathUtils.lerp(colAttr.getX(i), br, w),
          THREE.MathUtils.lerp(colAttr.getY(i), bg, w),
          THREE.MathUtils.lerp(colAttr.getZ(i), bb, w)
        );
        hsvAttr.setXYZ(
          i,
          THREE.MathUtils.lerp(hsvAttr.getX(i), 0, w),
          THREE.MathUtils.lerp(hsvAttr.getY(i), 0, w),
          THREE.MathUtils.lerp(hsvAttr.getZ(i), 0, w)
        );
      } else if (settings.mode === 'hueShift') {
        const jNoise = jitterDeg > 0.5 ? (hashVertex3(_vPos.x, _vPos.y, _vPos.z, seed) - 0.5) * 2 * (jitterDeg / 360) : 0;
        const targetHue = hueNorm + jNoise;
        hsvAttr.setXYZ(
          i,
          THREE.MathUtils.lerp(hsvAttr.getX(i), targetHue, w),
          THREE.MathUtils.lerp(hsvAttr.getY(i), satShift, w),
          THREE.MathUtils.lerp(hsvAttr.getZ(i), valShift, w)
        );
      } else if (settings.mode === 'organicJitter') {
        const spatial = organicSpatialNoise(_vPos.x, _vPos.y, _vPos.z, seed);
        const micro = (hashVertex3(_vPos.x, _vPos.y, _vPos.z, seed + 31) - 0.5) * 0.45;
        const patchNoise = THREE.MathUtils.clamp(spatial * 0.8 + micro, -1, 1);
        const effJitter = Math.max(18, jitterDeg);
        const hShiftLocal = hueNorm + patchNoise * (effJitter / 360);
        const sShiftLocal = satShift + patchNoise * 0.14;
        const vShiftLocal = valShift + patchNoise * 0.16;

        // Combine rich per-vertex HSV texture shift + subtle organic color tinting
        hsvAttr.setXYZ(
          i,
          THREE.MathUtils.lerp(hsvAttr.getX(i), hShiftLocal, w * 0.9),
          THREE.MathUtils.lerp(hsvAttr.getY(i), THREE.MathUtils.clamp(sShiftLocal, -0.8, 0.8), w * 0.85),
          THREE.MathUtils.lerp(hsvAttr.getZ(i), THREE.MathUtils.clamp(vShiftLocal, -0.5, 0.5), w * 0.85)
        );

        _cTarget.setHSL(
          (((baseH + patchNoise * (effJitter / 360)) % 1) + 1) % 1,
          THREE.MathUtils.clamp(baseS * 0.85, 0, 1),
          THREE.MathUtils.clamp(baseL + patchNoise * 0.1, 0.12, 0.94),
          THREE.SRGBColorSpace
        );
        _cCurr.setRGB(colAttr.getX(i), colAttr.getY(i), colAttr.getZ(i));
        _cCurr.getRGB(_srgbA, THREE.SRGBColorSpace);
        _cTarget.getRGB(_srgbB, THREE.SRGBColorSpace);
        const tintW = w * 0.42;
        _cTmp.setRGB(
          _srgbA.r + (_srgbB.r - _srgbA.r) * tintW,
          _srgbA.g + (_srgbB.g - _srgbA.g) * tintW,
          _srgbA.b + (_srgbB.b - _srgbA.b) * tintW,
          THREE.SRGBColorSpace
        );
        colAttr.setXYZ(i, _cTmp.r, _cTmp.g, _cTmp.b);
      } else {
        // 'paint': direct color variation painting (with hue shift & optional organic jitter!)
        const jOffset =
          jitterDeg > 0.5
            ? organicSpatialNoise(_vPos.x, _vPos.y, _vPos.z, seed) * (jitterDeg / 360)
            : 0;
        const vH = (((baseH + jOffset) % 1) + 1) % 1;
        _cTarget.setHSL(vH, baseS, baseL, THREE.SRGBColorSpace);

        _cCurr.setRGB(colAttr.getX(i), colAttr.getY(i), colAttr.getZ(i));
        _cCurr.getRGB(_srgbA, THREE.SRGBColorSpace);
        _cTarget.getRGB(_srgbB, THREE.SRGBColorSpace);
        _cTmp.setRGB(
          _srgbA.r + (_srgbB.r - _srgbA.r) * w,
          _srgbA.g + (_srgbB.g - _srgbA.g) * w,
          _srgbA.b + (_srgbB.b - _srgbA.b) * w,
          THREE.SRGBColorSpace
        );
        colAttr.setXYZ(i, _cTmp.r, _cTmp.g, _cTmp.b);

        // Gently neutralize prior HSV shift on painted area so chosen color reads true
        hsvAttr.setXYZ(
          i,
          THREE.MathUtils.lerp(hsvAttr.getX(i), 0, w * 0.65),
          THREE.MathUtils.lerp(hsvAttr.getY(i), 0, w * 0.65),
          THREE.MathUtils.lerp(hsvAttr.getZ(i), 0, w * 0.65)
        );
      }
    }

    if (meshTouched) {
      colAttr.needsUpdate = true;
      hsvAttr.needsUpdate = true;
      anyModified = true;
    }
  }

  return anyModified;
}

/**
 * Deterministically replays all saved 3D vertex paint strokes onto a built tree or object mesh.
 */
export function applyVertexPaintStrokesToObject(
  root: THREE.Object3D,
  strokes: VertexPaintStroke[] | undefined
): void {
  if (!strokes || strokes.length === 0) return;
  const meshes = prepareObjectForVertexPainting(root);
  if (meshes.length === 0) return;

  const pt = new THREE.Vector3();
  for (let s = 0; s < strokes.length; s++) {
    const stroke = strokes[s];
    const settings: VertexPaintBrushSettings = {
      radius: stroke.radius,
      opacity: stroke.opacity,
      softness: stroke.softness ?? 0.65,
      mode: stroke.mode,
      target: stroke.target,
      color: stroke.color,
      hueShift: stroke.hueShift,
      saturationShift: stroke.saturationShift ?? 0,
      valueShift: stroke.valueShift ?? 0,
      hueJitter: stroke.hueJitter ?? 0,
    };
    for (let p = 0; p < stroke.points.length; p++) {
      const [x, y, z] = stroke.points[p];
      pt.set(x, y, z);
      applyVertexPaintDab(root, pt, settings, stroke.seed, meshes);
    }
  }
}

/**
 * Generates an organic multi-dab auto-dapple stroke over the visible leaves or branches
 * so creators can immediately introduce painterly non-repetitive color variation in 1 click.
 */
export function generateOrganicDappleStroke(
  root: THREE.Object3D,
  settings: VertexPaintBrushSettings,
  seed: number
): VertexPaintStroke | null {
  root.updateMatrixWorld(true);
  const invRoot = root.matrixWorld.clone().invert();
  const candidates: THREE.Vector3[] = [];

  root.traverse((child) => {
    if (!(child instanceof THREE.Mesh) || child.userData.noVertexPaint || !child.visible) return;
    if (!matchesTarget(child, settings.target)) return;
    const pos = child.geometry.attributes.position as THREE.BufferAttribute | undefined;
    if (!pos || pos.count === 0) return;
    const mToR = new THREE.Matrix4().multiplyMatrices(invRoot, child.matrixWorld);
    const step = Math.max(1, Math.floor(pos.count / 18));
    for (let i = 0; i < pos.count; i += step) {
      candidates.push(new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i)).applyMatrix4(mToR));
    }
  });

  if (candidates.length === 0) return null;

  // Deterministic pseudo-random pick of ~14 organic dapple anchors
  let st = (seed >>> 0) || 1;
  const rand = () => {
    st = (st + 0x6d2b79f5) >>> 0;
    let t = Math.imul(st ^ (st >>> 15), 1 | st);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const count = Math.min(16, Math.max(6, Math.round(candidates.length * 0.25)));
  const points: [number, number, number][] = [];
  for (let i = 0; i < count; i++) {
    const pick = candidates[Math.floor(rand() * candidates.length)];
    points.push([
      Number(pick.x.toFixed(3)),
      Number(pick.y.toFixed(3)),
      Number(pick.z.toFixed(3)),
    ]);
  }

  return {
    id: `vp_${Date.now().toString(36)}_${Math.floor(rand() * 9999)}`,
    points,
    radius: Math.max(0.45, settings.radius * 1.15),
    opacity: Math.min(0.65, Math.max(0.28, settings.opacity)),
    softness: 0.82,
    mode: settings.mode === 'erase' ? 'organicJitter' : settings.mode,
    target: settings.target,
    color: settings.color,
    hueShift: settings.hueShift,
    saturationShift: settings.saturationShift,
    valueShift: settings.valueShift,
    hueJitter: Math.max(24, settings.hueJitter),
    seed,
  };
}
