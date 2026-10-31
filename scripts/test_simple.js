const fs = require('fs');
const THREE = require('three');

const glb = fs.readFileSync('apps/web/public/models/sign-avatar/sign-avatar.glb');
const jsonLen = glb.readUInt32LE(12);
const data = JSON.parse(glb.slice(20, 20 + jsonLen).toString('utf8'));
const bin = glb.slice(20 + jsonLen + 8);

function readAccessor(idx) {
  const acc = data.accessors[idx];
  const bv = data.bufferViews[acc.bufferView];
  const off = (bv.byteOffset || 0) + (acc.byteOffset || 0);
  const n = acc.type === 'VEC4' ? 4 : (acc.type === 'VEC3' ? 3 : 1);
  return new Float32Array(bin.buffer, bin.byteOffset + off, acc.count * n);
}

const cleanBoneName = (name) => {
  return name.replace(/^([a-zA-Z0-9_-]+:)?mixamorig\d*:?/i, '').replace(/^[a-zA-Z0-9_-]+:/, '');
};

// Build scene
const nodeMap = new Map();
data.nodes.forEach((n, i) => {
  const b = new THREE.Bone();
  b.name = cleanBoneName(n.name);
  nodeMap.set(i, b);
});
data.nodes.forEach((n, i) => {
  const parent = nodeMap.get(i);
  if (n.children) {
    n.children.forEach(cIdx => parent.add(nodeMap.get(cIdx)));
  }
});
const root = nodeMap.get(71); // Armature node

// Build all native clips
const nativeClips = [];
data.animations.forEach(anim => {
  const tracks = [];
  anim.channels.forEach(ch => {
    const rawNodeName = data.nodes[ch.target.node].name;
    const nodeName = cleanBoneName(rawNodeName);
    const sampler = anim.samplers[ch.sampler];
    const times = readAccessor(sampler.input);
    const values = readAccessor(sampler.output);
    if (ch.target.path === 'rotation') {
      tracks.push(new THREE.QuaternionKeyframeTrack(nodeName + '.quaternion', times, values));
    } else if (ch.target.path === 'translation') {
      tracks.push(new THREE.VectorKeyframeTrack(nodeName + '.position', times, values));
    } else if (ch.target.path === 'scale') {
      tracks.push(new THREE.VectorKeyframeTrack(nodeName + '.scale', times, values));
    }
  });
  let maxTime = 0;
  tracks.forEach(t => {
    const lastT = t.times[t.times.length - 1];
    if (lastT > maxTime) maxTime = lastT;
  });
  nativeClips.push(new THREE.AnimationClip(anim.name, maxTime, tracks));
});

console.log('Created native clips:', nativeClips.length);

const mixer = new THREE.AnimationMixer(root);

let totalTracks = 0;
let totalBound = 0;

nativeClips.forEach(clip => {
  const action = mixer.clipAction(clip);
  action.play();
  mixer.update(0.01);
  const bound = action._propertyBindings.filter(b => b && b.binding && b.binding.node).length;
  totalTracks += clip.tracks.length;
  totalBound += bound;
  if (bound !== clip.tracks.length) {
    console.warn(`Clip ${clip.name}: ${bound}/${clip.tracks.length} bound!`);
  }
});

console.log(`Native clips binding test: ${totalBound}/${totalTracks} tracks bound across ${nativeClips.length} clips!`);
