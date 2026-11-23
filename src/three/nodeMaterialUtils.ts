import * as THREE from 'three';
import {
  MeshBasicNodeMaterial,
  MeshLambertNodeMaterial,
  MeshPhongNodeMaterial,
  MeshPhysicalNodeMaterial,
  MeshStandardNodeMaterial,
  MeshToonNodeMaterial,
  NodeMaterialInstance,
  type NodeMaterial,
} from 'three/webgpu';
import {
  Fn,
  If,
  abs,
  clamp,
  dot,
  float,
  fract,
  materialColor,
  materialDiffuseRoughness,
  materialRoughness,
  materialReference,
  max,
  min,
  mix,
  normalWorld,
  positionWorld,
  pow,
  sin,
  cos,
  smoothstep,
  step,
  vec3,
  vec4,
} from 'three/tsl';
import type { EntityMaterialVariant, VisualStyleId, WeatheringPreset } from '../types/maker';
import { attachBotwPainterlyShader } from './botwStyleUtils';

export const WEATHERING_TYPE_MAP: Record<WeatheringPreset, number> = {
  none: 0,
  moss: 1,
  snow: 2,
  wet: 3,
};

// Ensure r187's NodeMaterial.setDefaultValues + copy() never triggers the deprecated
// `iridescenceThicknessRange` getter/setter warning on MeshPhysicalNodeMaterial clones.
let _physicalProtoCleaned = false;
function ensurePhysicalNodeProtoCleaned(): void {
  if (_physicalProtoCleaned) return;
  _physicalProtoCleaned = true;
  try {
    const proto = MeshPhysicalNodeMaterial.prototype as unknown as {
      iridescenceThicknessMinimum?: number;
      iridescenceThicknessMaximum?: number;
    };
    const probe = new MeshPhysicalNodeMaterial();
    probe.dispose();
    Object.defineProperty(proto, 'iridescenceThicknessRange', {
      configurable: true,
      enumerable: false,
      get(this: { iridescenceThicknessMinimum?: number; iridescenceThicknessMaximum?: number }) {
        return [this.iridescenceThicknessMinimum ?? 100, this.iridescenceThicknessMaximum ?? 400];
      },
      set(
        this: { iridescenceThicknessMinimum?: number; iridescenceThicknessMaximum?: number },
        value: unknown
      ) {
        if (Array.isArray(value) && value.length >= 2) {
          this.iridescenceThicknessMinimum = Number(value[0]) || 100;
          this.iridescenceThicknessMaximum = Number(value[1]) || 400;
        }
      },
    });
  } catch {
    // ignore if prototype is sealed
  }
}
ensurePhysicalNodeProtoCleaned();

export function isNodeMaterialInstance(mat: THREE.Material | null | undefined): mat is NodeMaterial {
  return Boolean(mat && (mat as unknown as { isNodeMaterial?: boolean }).isNodeMaterial);
}

export function isRealNodeMaterialInstance(
  mat: THREE.Material | null | undefined
): mat is NodeMaterialInstance {
  return Boolean(
    mat && (mat as unknown as { isNodeMaterialInstance?: boolean }).isNodeMaterialInstance
  );
}

/**
 * Enables r187's Energy-Preserving Oren-Nayar (EON) `diffuseRoughness` on matte PBR materials
 * by upgrading classic `MeshStandardMaterial` / `MeshStandardNodeMaterial` into
 * `MeshPhysicalNodeMaterial` (ignoring celToon, transparent, or unlit materials).
 */
export function enableDiffuseRoughnessEon(
  mat: THREE.Material,
  diffuseRoughnessVal = 0.88
): THREE.Material {
  if ((mat as THREE.MeshToonMaterial).isMeshToonMaterial) return mat;
  if ((mat as THREE.MeshBasicMaterial).isMeshBasicMaterial) return mat;
  if (mat.transparent) return mat;

  if ((mat as MeshPhysicalNodeMaterial).isMeshPhysicalNodeMaterial) {
    (mat as MeshPhysicalNodeMaterial).diffuseRoughness = THREE.MathUtils.clamp(
      diffuseRoughnessVal,
      0,
      1
    );
    return mat;
  }

  if (isRealNodeMaterialInstance(mat)) {
    const asPhys = mat as unknown as { diffuseRoughness?: number };
    asPhys.diffuseRoughness = THREE.MathUtils.clamp(diffuseRoughnessVal, 0, 1);
    return mat;
  }

  const phys = new MeshPhysicalNodeMaterial();
  const src = mat as THREE.MeshStandardMaterial & { diffuseRoughness?: number };
  if (src.color) phys.color.copy(src.color);
  phys.map = src.map ?? null;
  phys.alphaMap = src.alphaMap ?? null;
  phys.normalMap = src.normalMap ?? null;
  if (src.normalScale) phys.normalScale.copy(src.normalScale);
  phys.bumpMap = src.bumpMap ?? null;
  phys.bumpScale = src.bumpScale ?? 1;
  phys.roughness = src.roughness ?? 0.86;
  phys.metalness = src.metalness ?? 0;
  phys.diffuseRoughness = THREE.MathUtils.clamp(
    src.diffuseRoughness ?? diffuseRoughnessVal,
    0,
    1
  );
  phys.roughnessMap = src.roughnessMap ?? null;
  phys.metalnessMap = src.metalnessMap ?? null;
  if (src.emissive) phys.emissive.copy(src.emissive);
  phys.emissiveIntensity = src.emissiveIntensity ?? 1;
  phys.emissiveMap = src.emissiveMap ?? null;
  phys.aoMap = src.aoMap ?? null;
  phys.aoMapIntensity = src.aoMapIntensity ?? 1;
  phys.envMapIntensity = src.envMapIntensity ?? 1;

  phys.name = mat.name;
  phys.transparent = mat.transparent;
  phys.opacity = mat.opacity;
  phys.alphaTest = mat.alphaTest;
  phys.side = mat.side;
  phys.depthWrite = mat.depthWrite;
  phys.depthTest = mat.depthTest;
  phys.blending = mat.blending;
  phys.vertexColors = mat.vertexColors;
  phys.fog = src.fog ?? true;
  phys.toneMapped = mat.toneMapped;
  phys.wireframe = Boolean(src.wireframe);
  (phys as unknown as { flatShading?: boolean }).flatShading = Boolean(src.flatShading);
  phys.userData = { ...mat.userData };
  phys.needsUpdate = true;

  if (mat !== phys) {
    mat.dispose();
  }
  return phys;
}

/**
 * Upgrades any classic Three.js Material (MeshStandardMaterial, MeshToonMaterial, etc.)
 * into its r187 NodeMaterial equivalent while preserving all surface maps, PBR/Toon params,
 * EON diffuseRoughness, blending, transparency, and vertexColors flags so TSL hooks work
 * natively under WebGPURenderer.
 */
export function upgradeToNodeMaterial(mat: THREE.Material): NodeMaterial {
  if (isNodeMaterialInstance(mat)) {
    return mat;
  }

  let target: NodeMaterial;
  const src = mat as THREE.MeshStandardMaterial &
    THREE.MeshToonMaterial &
    THREE.MeshPhysicalMaterial &
    THREE.MeshPhongMaterial & { diffuseRoughness?: number };

  if ((mat as THREE.MeshToonMaterial).isMeshToonMaterial) {
    const toon = new MeshToonNodeMaterial();
    toon.color.copy(src.color ?? new THREE.Color('#ffffff'));
    toon.map = src.map ?? null;
    toon.alphaMap = src.alphaMap ?? null;
    toon.normalMap = src.normalMap ?? null;
    if (src.normalScale) toon.normalScale.copy(src.normalScale);
    toon.bumpMap = src.bumpMap ?? null;
    toon.bumpScale = src.bumpScale ?? 1;
    toon.emissive.copy(src.emissive ?? new THREE.Color('#000000'));
    toon.emissiveIntensity = src.emissiveIntensity ?? 1;
    toon.emissiveMap = src.emissiveMap ?? null;
    toon.aoMap = src.aoMap ?? null;
    toon.aoMapIntensity = src.aoMapIntensity ?? 1;
    toon.gradientMap = src.gradientMap ?? null;
    target = toon;
  } else if ((mat as THREE.MeshPhysicalMaterial).isMeshPhysicalMaterial) {
    const phys = new MeshPhysicalNodeMaterial();
    phys.color.copy(src.color ?? new THREE.Color('#ffffff'));
    phys.map = src.map ?? null;
    phys.alphaMap = src.alphaMap ?? null;
    phys.normalMap = src.normalMap ?? null;
    if (src.normalScale) phys.normalScale.copy(src.normalScale);
    phys.bumpMap = src.bumpMap ?? null;
    phys.bumpScale = src.bumpScale ?? 1;
    phys.roughness = src.roughness ?? 1;
    phys.metalness = src.metalness ?? 0;
    phys.diffuseRoughness = src.diffuseRoughness ?? 0;
    phys.roughnessMap = src.roughnessMap ?? null;
    phys.metalnessMap = src.metalnessMap ?? null;
    phys.emissive.copy(src.emissive ?? new THREE.Color('#000000'));
    phys.emissiveIntensity = src.emissiveIntensity ?? 1;
    phys.emissiveMap = src.emissiveMap ?? null;
    phys.aoMap = src.aoMap ?? null;
    phys.aoMapIntensity = src.aoMapIntensity ?? 1;
    phys.envMapIntensity = src.envMapIntensity ?? 1;
    phys.clearcoat = src.clearcoat ?? 0;
    phys.clearcoatRoughness = src.clearcoatRoughness ?? 0;
    phys.ior = src.ior ?? 1.5;
    phys.iridescence = src.iridescence ?? 0;
    phys.iridescenceIOR = src.iridescenceIOR ?? 1.3;
    phys.iridescenceThicknessMinimum = src.iridescenceThicknessMinimum ?? 100;
    phys.iridescenceThicknessMaximum = src.iridescenceThicknessMaximum ?? 400;
    target = phys;
  } else if ((mat as THREE.MeshBasicMaterial).isMeshBasicMaterial) {
    const basic = new MeshBasicNodeMaterial();
    basic.color.copy(src.color ?? new THREE.Color('#ffffff'));
    basic.map = src.map ?? null;
    basic.alphaMap = src.alphaMap ?? null;
    basic.aoMap = src.aoMap ?? null;
    basic.aoMapIntensity = src.aoMapIntensity ?? 1;
    target = basic;
  } else if ((mat as THREE.MeshLambertMaterial).isMeshLambertMaterial) {
    const lambert = new MeshLambertNodeMaterial();
    lambert.color.copy(src.color ?? new THREE.Color('#ffffff'));
    lambert.map = src.map ?? null;
    lambert.alphaMap = src.alphaMap ?? null;
    lambert.emissive.copy(src.emissive ?? new THREE.Color('#000000'));
    lambert.emissiveIntensity = src.emissiveIntensity ?? 1;
    lambert.emissiveMap = src.emissiveMap ?? null;
    target = lambert;
  } else if ((mat as THREE.MeshPhongMaterial).isMeshPhongMaterial) {
    const phong = new MeshPhongNodeMaterial();
    phong.color.copy(src.color ?? new THREE.Color('#ffffff'));
    phong.map = src.map ?? null;
    phong.alphaMap = src.alphaMap ?? null;
    phong.normalMap = src.normalMap ?? null;
    if (src.normalScale) phong.normalScale.copy(src.normalScale);
    phong.bumpMap = src.bumpMap ?? null;
    phong.bumpScale = src.bumpScale ?? 1;
    phong.emissive.copy(src.emissive ?? new THREE.Color('#000000'));
    phong.emissiveIntensity = src.emissiveIntensity ?? 1;
    phong.emissiveMap = src.emissiveMap ?? null;
    phong.shininess = src.shininess ?? 30;
    target = phong;
  } else {
    const std = new MeshStandardNodeMaterial();
    if (src.color) std.color.copy(src.color);
    std.map = src.map ?? null;
    std.alphaMap = src.alphaMap ?? null;
    std.normalMap = src.normalMap ?? null;
    if (src.normalScale) std.normalScale.copy(src.normalScale);
    std.bumpMap = src.bumpMap ?? null;
    std.bumpScale = src.bumpScale ?? 1;
    std.roughness = src.roughness ?? 1;
    std.metalness = src.metalness ?? 0;
    std.roughnessMap = src.roughnessMap ?? null;
    std.metalnessMap = src.metalnessMap ?? null;
    if (src.emissive) std.emissive.copy(src.emissive);
    std.emissiveIntensity = src.emissiveIntensity ?? 1;
    std.emissiveMap = src.emissiveMap ?? null;
    std.aoMap = src.aoMap ?? null;
    std.aoMapIntensity = src.aoMapIntensity ?? 1;
    std.envMapIntensity = src.envMapIntensity ?? 1;
    target = std;
  }

  target.name = mat.name;
  target.transparent = mat.transparent;
  target.opacity = mat.opacity;
  target.alphaTest = mat.alphaTest;
  target.side = mat.side;
  target.depthWrite = mat.depthWrite;
  target.depthTest = mat.depthTest;
  target.blending = mat.blending;
  target.vertexColors = mat.vertexColors;
  target.fog = (mat as THREE.MeshStandardMaterial).fog ?? true;
  target.toneMapped = mat.toneMapped;
  const targetStd = target as unknown as THREE.MeshStandardMaterial;
  targetStd.wireframe = (mat as THREE.MeshStandardMaterial).wireframe ?? false;
  targetStd.flatShading = Boolean((mat as THREE.MeshStandardMaterial).flatShading);
  target.userData = { ...mat.userData };
  target.needsUpdate = true;

  return target;
}

/**
 * Shared TSL perceptual linear-RGB <-> sRGB <-> HSV shift helper
 * (matches the original GLSL hue/saturation/value shifts across Canopy Engine and Vertex Painter).
 */
export const applyHsvShiftLinearTsl = Fn(([rgbLinear, hsvShift]: [any, any]) => {
  const outCol = vec3(rgbLinear.rgb).toVar('hsvOutRgb');
  const mag = abs(hsvShift.x).add(abs(hsvShift.y)).add(abs(hsvShift.z));

  If(mag.greaterThan(0.0005), () => {
    const srgbCol = pow(max(outCol, vec3(0.0001)), vec3(1.0 / 2.2));
    const K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    const p = mix(vec4(srgbCol.bg, K.wz), vec4(srgbCol.gb, K.xy), step(srgbCol.b, srgbCol.g));
    const q = mix(vec4(p.xyw, srgbCol.r), vec4(srgbCol.r, p.yzx), step(p.x, srgbCol.r));
    const d = q.x.sub(min(q.w, q.y));
    const e = float(1.0e-10);
    const hue = abs(q.z.add(q.w.sub(q.y).div(d.mul(6.0).add(e))));
    const sat = d.div(q.x.add(e));
    const val = q.x;

    const newH = fract(hue.add(hsvShift.x));
    const newS = clamp(sat.add(hsvShift.y), 0.0, 1.0);
    const newV = clamp(val.mul(float(1.0).add(hsvShift.z)), 0.0, 1.4);

    const K2 = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    const p2 = abs(fract(vec3(newH, newH, newH).add(K2.xyz)).mul(6.0).sub(K2.www));
    const shiftedSrgb = clamp(mix(K2.xxx, clamp(p2.sub(K2.xxx), 0.0, 1.0), newS), 0.0, 1.4).mul(newV);

    outCol.assign(pow(max(shiftedSrgb, vec3(0.0)), vec3(2.2)));
  });

  return outCol;
});

const _defaultTintFallback = new THREE.Color('#ffffff');

export function safeMaterialReference<T>(property: string, inputType: string, fallback: T) {
  const node = materialReference(property, inputType) as unknown as {
    reference?: Record<string, unknown> | null;
    properties?: string[];
    getValueFromReference?: (obj?: Record<string, unknown> | null) => unknown;
  };
  node.getValueFromReference = function (object = node.reference) {
    if (!object) return fallback;
    const props = node.properties ?? property.split('.');
    let cur: unknown = object;
    for (let i = 0; i < props.length; i++) {
      if (cur === null || cur === undefined || typeof cur !== 'object') {
        return fallback;
      }
      cur = (cur as Record<string, unknown>)[props[i]];
    }
    return cur === undefined || cur === null ? fallback : cur;
  };
  return node;
}

/**
 * Reusable TSL Weathering shader fragment (`moss`, `snow`, `wet`, `none`) driven purely by
 * safe `materialReference('userData.uWeatherType', 'float')` and `materialReference('userData.uWeatherIntensity', 'float')`
 * so changing presets on `NodeMaterialInstance` updates uniforms in real time with ZERO recompilation,
 * and internal override materials (like ShadowMaterial) cleanly fall back to safe defaults.
 */
export function buildWeatheringColorTsl(baseRgbNode: unknown) {
  return Fn(() => {
    const uTintColor = safeMaterialReference('userData.uTintColor', 'color', _defaultTintFallback) as unknown as ReturnType<typeof vec3>;
    const uTintMix = safeMaterialReference('userData.uTintMix', 'float', 0) as unknown as ReturnType<typeof float>;
    const uWeatherType = safeMaterialReference('userData.uWeatherType', 'float', 0) as unknown as ReturnType<typeof float>;
    const uWeatherIntensity = safeMaterialReference('userData.uWeatherIntensity', 'float', 0) as unknown as ReturnType<typeof float>;

    const wp = positionWorld;
    const wnY = clamp(normalWorld.y, -1.0, 1.0);
    const upFacing = smoothstep(0.12, 0.72, wnY);

    // Deterministic micro noise in world space for organic moss/frost patches
    const nA = sin(wp.x.mul(2.1).add(wp.z.mul(1.7))).mul(cos(wp.z.mul(2.4).sub(wp.y.mul(1.3))));
    const nB = cos(wp.x.mul(5.3).sub(wp.z.mul(4.1))).mul(0.5);
    const patchNoise = smoothstep(-0.35, 0.55, nA.add(nB));

    const baseRgb = vec3(baseRgbNode as THREE.Vector3);
    const tinted = mix(baseRgb, baseRgb.mul(vec3(uTintColor)), clamp(uTintMix, 0.0, 1.0)).toVar();

    // Weather 1 = Moss (velvety woodland moss clinging onto upward and shadowed crevices)
    const isMoss = step(0.5, uWeatherType).mul(float(1.0).sub(step(1.5, uWeatherType)));
    const mossMask = clamp(upFacing.mul(0.75).add(0.25).mul(mix(0.45, 1.0, patchNoise)), 0.0, 1.0)
      .mul(uWeatherIntensity)
      .mul(isMoss);
    const mossCol = mix(vec3(0.16, 0.38, 0.14), vec3(0.28, 0.52, 0.2), patchNoise);
    tinted.assign(mix(tinted, mossCol, clamp(mossMask.mul(0.88), 0.0, 0.92)));

    // Weather 2 = Snow (crisp frosty mantle on upper surfaces)
    const isSnow = step(1.5, uWeatherType).mul(float(1.0).sub(step(2.5, uWeatherType)));
    const snowMask = smoothstep(0.18, 0.62, wnY.add(patchNoise.sub(0.5).mul(0.28)))
      .mul(uWeatherIntensity)
      .mul(isSnow);
    const snowCol = mix(vec3(0.82, 0.89, 0.96), vec3(0.96, 0.98, 1.0), upFacing);
    tinted.assign(mix(tinted, snowCol, clamp(snowMask.mul(0.95), 0.0, 0.96)));

    // Weather 3 = Wet (darkened albedo saturation as if rained on)
    const isWet = step(2.5, uWeatherType);
    const wetMask = clamp(uWeatherIntensity.mul(0.85), 0.0, 0.95).mul(isWet);
    tinted.assign(mix(tinted, tinted.mul(0.56), wetMask));

    return tinted;
  })();
}

export function buildWeatheringRoughnessTsl() {
  return Fn(() => {
    const uWeatherType = safeMaterialReference('userData.uWeatherType', 'float', 0) as unknown as ReturnType<typeof float>;
    const uWeatherIntensity = safeMaterialReference('userData.uWeatherIntensity', 'float', 0) as unknown as ReturnType<typeof float>;
    const baseRough = materialRoughness;
    const wnY = clamp(normalWorld.y, -1.0, 1.0);
    const upFacing = smoothstep(0.18, 0.68, wnY);

    const isMoss = step(0.5, uWeatherType).mul(float(1.0).sub(step(1.5, uWeatherType)));
    const isSnow = step(1.5, uWeatherType).mul(float(1.0).sub(step(2.5, uWeatherType)));
    const isWet = step(2.5, uWeatherType);

    const r = baseRough.toVar();
    r.assign(mix(r, max(r, 0.9), isMoss.mul(uWeatherIntensity).mul(upFacing)));
    r.assign(mix(r, float(0.45), isSnow.mul(uWeatherIntensity).mul(upFacing).mul(0.8)));
    r.assign(mix(r, float(0.11), isWet.mul(clamp(uWeatherIntensity.mul(0.95), 0.0, 0.95))));
    return clamp(r, 0.05, 1.0);
  })();
}

export function applyWeatheringUserData(
  targetUserData: Record<string, unknown>,
  opts?: {
    tintHex?: string;
    tintMix?: number;
    weathering?: WeatheringPreset;
    weatheringIntensity?: number;
  }
) {
  if (!(targetUserData.uTintColor instanceof THREE.Color)) {
    targetUserData.uTintColor = new THREE.Color('#ffffff');
  }
  if (opts?.tintHex) {
    (targetUserData.uTintColor as THREE.Color).set(opts.tintHex);
  } else if (opts && 'tintHex' in opts && opts.tintHex === undefined) {
    (targetUserData.uTintColor as THREE.Color).set('#ffffff');
  }
  if (typeof opts?.tintMix === 'number') {
    targetUserData.uTintMix = THREE.MathUtils.clamp(opts.tintMix, 0, 1);
  } else if (typeof targetUserData.uTintMix !== 'number') {
    targetUserData.uTintMix = 0;
  }
  if (opts?.weathering !== undefined) {
    targetUserData.uWeatherType = WEATHERING_TYPE_MAP[opts.weathering] ?? 0;
  } else if (typeof targetUserData.uWeatherType !== 'number') {
    targetUserData.uWeatherType = 0;
  }
  if (typeof opts?.weatheringIntensity === 'number') {
    targetUserData.uWeatherIntensity = THREE.MathUtils.clamp(opts.weatheringIntensity, 0, 1);
  } else if (typeof targetUserData.uWeatherIntensity !== 'number') {
    targetUserData.uWeatherIntensity = 0.65;
  }
}

export interface VariantBaseOptions {
  style?: VisualStyleId;
  hasMap?: boolean;
  doubleSide?: boolean;
  transparent?: boolean;
  vertexColors?: boolean;
  diffuseRoughness?: number;
  roughness?: number;
  metalness?: number;
}

const variantBaseCache = new Map<string, MeshPhysicalNodeMaterial>();

/**
 * Returns a shared compiled `MeshPhysicalNodeMaterial` (EON-capable + TSL weathering/tint hooks)
 * keyed only by pipeline structural flags so creating/tweaking dozens of `NodeMaterialInstance`
 * variants compiles ZERO additional shaders!
 */
export function getSharedWeatheringBaseMaterial(opts: VariantBaseOptions = {}): MeshPhysicalNodeMaterial {
  const key = [
    opts.doubleSide ? 'ds' : 'fs',
    opts.transparent ? 'tr' : 'op',
    opts.vertexColors ? 'vc' : 'nvc',
    opts.hasMap ? 'map' : 'nomap',
  ].join('|');

  const cached = variantBaseCache.get(key);
  if (cached) return cached;

  const base = new MeshPhysicalNodeMaterial({
    color: '#ffffff',
    roughness: opts.roughness ?? 0.84,
    metalness: opts.metalness ?? 0.02,
    diffuseRoughness: opts.diffuseRoughness ?? 0.72,
    side: opts.doubleSide ? THREE.DoubleSide : THREE.FrontSide,
    transparent: Boolean(opts.transparent),
    vertexColors: Boolean(opts.vertexColors),
  });

  // `materialColor` automatically reads `material.color` and multiplies `material.map` via `MaterialNode`
  base.colorNode = buildWeatheringColorTsl(vec3(materialColor));
  base.roughnessNode = buildWeatheringRoughnessTsl();
  base.diffuseRoughnessNode = materialDiffuseRoughness;
  attachBotwPainterlyShader(base, { rimStrengthScale: 0.85, shadowLiftScale: 0.85 });

  applyWeatheringUserData(base.userData, {
    tintHex: '#ffffff',
    tintMix: 0,
    weathering: 'none',
    weatheringIntensity: 0.65,
  });
  base.needsUpdate = true;
  variantBaseCache.set(key, base);
  return base;
}

/**
 * Creates a `NodeMaterialInstance` backed by `baseNodeMat` with cleanly cloned mutable
 * Color / Vector2 / userData properties (preventing shared Color references across instances).
 */
export function createMaterialVariantInstance(
  baseNodeMat: NodeMaterial,
  init?: {
    colorHex?: string;
    tintHex?: string;
    tintMix?: number;
    roughness?: number;
    metalness?: number;
    diffuseRoughness?: number;
    weathering?: WeatheringPreset;
    weatheringIntensity?: number;
    map?: THREE.Texture | null;
    normalMap?: THREE.Texture | null;
    normalScale?: number;
    bumpMap?: THREE.Texture | null;
    bumpScale?: number;
    opacity?: number;
    transparent?: boolean;
  }
): NodeMaterialInstance {
  const inst = new NodeMaterialInstance(baseNodeMat);
  const instAny = inst as unknown as THREE.MeshPhysicalMaterial & {
    diffuseRoughness?: number;
  };
  const baseAny = baseNodeMat as unknown as THREE.MeshPhysicalMaterial & {
    diffuseRoughness?: number;
  };

  instAny.color = baseAny.color ? baseAny.color.clone() : new THREE.Color('#ffffff');
  instAny.emissive = baseAny.emissive ? baseAny.emissive.clone() : new THREE.Color('#000000');
  instAny.normalScale = baseAny.normalScale ? baseAny.normalScale.clone() : new THREE.Vector2(1, 1);

  if (init?.colorHex) instAny.color.set(init.colorHex);
  if (typeof init?.roughness === 'number') instAny.roughness = init.roughness;
  if (typeof init?.metalness === 'number') instAny.metalness = init.metalness;
  if (typeof init?.diffuseRoughness === 'number') instAny.diffuseRoughness = init.diffuseRoughness;
  if (init?.map !== undefined) instAny.map = init.map;
  if (init?.normalMap !== undefined) instAny.normalMap = init.normalMap;
  if (typeof init?.normalScale === 'number') {
    instAny.normalScale.set(init.normalScale, init.normalScale);
  }
  if (init?.bumpMap !== undefined) instAny.bumpMap = init.bumpMap;
  if (typeof init?.bumpScale === 'number') instAny.bumpScale = init.bumpScale;
  if (typeof init?.opacity === 'number') instAny.opacity = init.opacity;
  if (typeof init?.transparent === 'boolean') instAny.transparent = init.transparent;

  applyWeatheringUserData(inst.userData, {
    tintHex: init?.tintHex,
    tintMix: init?.tintMix,
    weathering: init?.weathering,
    weatheringIntensity: init?.weatheringIntensity,
  });
  return inst;
}

/**
 * Updates uniform values on a `NodeMaterialInstance` or concrete variant material in real time
 * with ZERO shader recompilation freezes!
 */
export function updateMaterialVariantInstance(
  mat: THREE.Material,
  variant: {
    colorHex?: string;
    tintHex?: string;
    tintMix?: number;
    roughness?: number;
    diffuseRoughness?: number;
    weathering?: WeatheringPreset;
    weatheringIntensity?: number;
  }
) {
  const m = mat as unknown as {
    color?: THREE.Color;
    roughness?: number;
    diffuseRoughness?: number;
    userData: Record<string, unknown>;
  };

  if (variant.colorHex && m.color) {
    m.color.set(variant.colorHex);
  }
  if (typeof variant.roughness === 'number' && typeof m.roughness === 'number') {
    m.roughness = THREE.MathUtils.clamp(variant.roughness, 0.05, 1);
  }
  if (typeof variant.diffuseRoughness === 'number') {
    m.diffuseRoughness = THREE.MathUtils.clamp(variant.diffuseRoughness, 0, 1);
  }

  applyWeatheringUserData(m.userData, {
    tintHex: variant.tintHex,
    tintMix: variant.tintMix,
    weathering: variant.weathering,
    weatheringIntensity: variant.weatheringIntensity,
  });
}

/**
 * Wraps every opaque/standard mesh inside an entity's 3D hierarchy into a shared-shader
 * `NodeMaterialInstance` so per-entity color, roughness, EON `diffuseRoughness`, and
 * weathering presets (`moss`, `snow`, `wet`) update live via uniforms without recompiling shaders!
 * Snapshot original defaults into `userData.__baseVariantDefaults` so resetting returns cleanly.
 */
export function applyEntityMaterialVariant(
  root: THREE.Object3D,
  variant?: EntityMaterialVariant,
  isMatteEntity = true
): void {
  const hasVariant = Boolean(
    variant &&
      (variant.colorHex ||
        typeof variant.roughness === 'number' ||
        typeof variant.diffuseRoughness === 'number' ||
        (variant.weathering && variant.weathering !== 'none'))
  );

  const wrappedCache = new Map<THREE.Material, THREE.Material>();
  const kept = new Set<THREE.Material>();
  const replaced = new Set<THREE.Material>();

  const wrap = (m: THREE.Material) => {
    const cached = wrappedCache.get(m);
    if (cached) return cached;
    const next = wrapSingleEntityMaterial(m, variant, isMatteEntity, hasVariant);
    wrappedCache.set(m, next);
    kept.add(next);
    if (next !== m) {
      replaced.add(m);
    }
    return next;
  };

  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;
    if (Array.isArray(mesh.material)) {
      mesh.material = mesh.material.map(wrap);
    } else if (mesh.material) {
      mesh.material = wrap(mesh.material);
    }
  });

  replaced.forEach((oldMat) => {
    if (!kept.has(oldMat)) {
      oldMat.dispose();
    }
  });
}

function wrapSingleEntityMaterial(
  mat: THREE.Material,
  variant: EntityMaterialVariant | undefined,
  isMatteEntity: boolean,
  hasVariant: boolean
): THREE.Material {
  // Never disturb custom particle / unlit glow materials or already-toon materials unless explicitly overridden
  if ((mat as THREE.MeshBasicMaterial).isMeshBasicMaterial && !hasVariant) return mat;

  // Already wrapped into our variant-aware NodeMaterialInstance (or concrete MeshPhysicalNodeMaterial for meshlets)
  if (mat.userData?.__variantWrapped) {
    applyVariantToWrappedMaterial(mat, variant);
    return mat;
  }

  const src = mat as THREE.MeshStandardMaterial & { diffuseRoughness?: number };
  const origColorHex = src.color ? `#${src.color.getHexString()}` : '#ffffff';
  const origRoughness = typeof src.roughness === 'number' ? src.roughness : 0.82;
  const origDiffuseRoughness =
    typeof src.diffuseRoughness === 'number' ? src.diffuseRoughness : isMatteEntity ? 0.84 : 0.25;

  // Only convert to NodeMaterialInstance when EON/variant is used or standard material benefits from EON + live edits
  if (!hasVariant && !isMatteEntity) {
    return mat;
  }

  const base = getSharedWeatheringBaseMaterial({
    hasMap: Boolean(src.map),
    doubleSide: mat.side === THREE.DoubleSide,
    transparent: mat.transparent,
    vertexColors: mat.vertexColors,
  });

  const inst = createMaterialVariantInstance(base, {
    colorHex: origColorHex,
    roughness: origRoughness,
    metalness: typeof src.metalness === 'number' ? src.metalness : 0.04,
    diffuseRoughness: origDiffuseRoughness,
    map: src.map ?? null,
    normalMap: src.normalMap ?? null,
    normalScale: src.normalScale ? src.normalScale.x : 1,
    bumpMap: src.bumpMap ?? null,
    bumpScale: src.bumpScale ?? 1,
    opacity: mat.opacity,
    transparent: mat.transparent,
    weathering: 'none',
    weatheringIntensity: 0.65,
  });

  inst.name = mat.name;
  inst.alphaTest = mat.alphaTest;
  inst.depthWrite = mat.depthWrite;
  inst.depthTest = mat.depthTest;
  (inst as unknown as { fog: boolean }).fog = src.fog ?? true;
  inst.toneMapped = mat.toneMapped;
  inst.userData.__variantWrapped = true;
  inst.userData.__baseVariantDefaults = {
    colorHex: origColorHex,
    roughness: origRoughness,
    diffuseRoughness: origDiffuseRoughness,
  };

  applyVariantToWrappedMaterial(inst, variant);
  return inst;
}

function applyVariantToWrappedMaterial(mat: THREE.Material, variant?: EntityMaterialVariant) {
  const defaults = (mat.userData?.__baseVariantDefaults as
    | { colorHex: string; roughness: number; diffuseRoughness: number }
    | undefined) ?? {
    colorHex: '#ffffff',
    roughness: 0.84,
    diffuseRoughness: 0.84,
  };

  const tintActive = Boolean(variant?.colorHex);
  updateMaterialVariantInstance(mat, {
    colorHex: variant?.colorHex ? '#ffffff' : defaults.colorHex,
    tintHex: variant?.colorHex ?? '#ffffff',
    tintMix: tintActive ? 0.82 : 0,
    roughness: typeof variant?.roughness === 'number' ? variant.roughness : defaults.roughness,
    diffuseRoughness:
      typeof variant?.diffuseRoughness === 'number'
        ? variant.diffuseRoughness
        : defaults.diffuseRoughness,
    weathering: variant?.weathering ?? 'none',
    weatheringIntensity: variant?.weatheringIntensity ?? 0.65,
  });
}

/**
 * Live zero-stutter in-place update when an entity's `materialVariant` changes in the inspector.
 */
export function updateEntityMaterialVariantInPlace(
  root: THREE.Object3D,
  variant?: EntityMaterialVariant,
  isMatteEntity = true
): void {
  applyEntityMaterialVariant(root, variant, isMatteEntity);
}

/**
 * Checks if two entities are structurally identical (so we don't tear down & rebuild 3D meshes
 * when only `materialVariant` or `meshletEnabled` is edited in real time).
 */
export function stripLiveVariantFieldsForStructKey(entities: unknown[]): unknown[] {
  return entities.map((e) => {
    if (!e || typeof e !== 'object') return e;
    const {
      materialVariant: _mv,
      meshletEnabled: _me,
      ...rest
    } = e as Record<string, unknown>;
    return rest;
  });
}
