# GLS' Experimental Ray Tracer

basically just throwing things at the wall and seeing what sticks.

## Roadmap

- Complex Lighting
  - [ ] Refraction
  - [ ] Caustics
  - [ ] Subsurface Scattering
  - [ ] PBR - Disney Model
  - [ ] Volumetrics
  - [ ] High density materials (Hair, Grass, Foliage, etc)
  - [ ] More accurate light transport models - Photon Mapping?
- Performance Tuning
  - [ ] Importance Sampling
  - [ ] GPU Compute via wgpu compute shaders
  - [ ] Benchmarking - Criterion
  - [ ] Faster BVH construction
  - [ ] Investigate build times for large models (Stanford Lucy - 5s to load GLTF, 7s to build BVH)
  - [ ] Meshlets - UE5 style
  - [ ] Dynamic Programming (World-space pixel caching)
  - [ ] Potential to lower FP precision
- Post Processing
  - [ ] HDR
  - [ ] Better Tonemapping
  - [ ] Fun shaders for stylization
- Misc
  - [ ] Real-time Interface
  - [ ] BVH caching
  - [ ] Animations
  - [ ] USD support
  - [ ] More complex camera modeling
