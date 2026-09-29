# OpenHawk

**OpenHawk: Physics-Based 3D Tennis Tracking from Broadcast Video**

OpenHawk turns ordinary single-camera tennis broadcast video into physically consistent 3D ball
flights, bounce and contact events, and player positions, fully automatically. Each point is
fitted jointly with a 3D flight and bounce model (drag, Magnus lift and spin, including sidespin),
and uncertain events are kept only when the physics supports them.

This repository will hold the open-source pipeline and the OpenHawk dataset of processed 3D
tracking data (3D flights, events, player positions and match metadata; no video or images).

**Status:** the code and data release is being prepared and will be published here.

## License

Apache License 2.0 (see `LICENSE`).

## Trademark note

Hawk-Eye is a registered trademark of Hawk-Eye Innovations Ltd. OpenHawk is an independent
project and is not affiliated with, sponsored by, or endorsed by Hawk-Eye Innovations.
