# Manifest

Exported from a private repository at commit `e84c6b62510d` as a fresh tree (no history). Each included path lists why it is here; excluded files are counted by rule.

Entry points: `cv.pipeline.product_runner`, `cv.validation.hawkeye_bounce_model`. 377 repository modules in the closure; 701 files included.

## Excluded (by rule)

| rule | files |
|---|---:|
| internal development notes and tool configuration | 159 |
| environment/secrets template | 1 |
| repository tooling not needed by the public tree | 3 |
| secondary downstream work, not in the video-to-3D path | 22 |
| research experiments not imported by the product path | 1890 |
| pipeline modules not reached from the product entry points | 133 |
| validation/labeling tools not imported by the product path | 556 |
| labels (see docs/EVALUATION.md: available on request) | 760 |
| private portal and review tools (only the 3D viewer and exporters used by the product path are included) | 143 |
| data directory links (not code) | 3 |
| third-party papers/articles (copyrighted; cite instead) | 4 |
| private operations tooling | 75 |

592 test files next to included modules are not exported because they import excluded code or replay private artifacts. A few text edits replaced absolute paths, host names and personal names with neutral ones.

## Replaced by public versions

- `cv/viz/portal_3d_src/build.py`: public version (private-only functionality removed)
- `cv/viz/portal_3d_src/test_build.py`: public version (private-only functionality removed)

## Included

| path | reason |
|---|---|
| `.gitignore` | authored for the public release (overlay) |
| `CITATION.cff` | authored for the public release (overlay) |
| `LICENSE` | authored for the public release (overlay) |
| `NOTICE` | authored for the public release (overlay) |
| `README.md` | authored for the public release (overlay) |
| `charting/tennis_charting/__init__.py` | product closure (import; first parent: cv.validation.ground_truth (sys.path bare import)) |
| `charting/tennis_charting/notation.py` | product closure (import; first parent: cv.validation.ground_truth (sys.path bare import)) |
| `conftest.py` | pytest options (slow physics fits behind --runslow) |
| `cv/experiments/ball_track_labeler/__init__.py` | product closure (import; first parent: package of cv.experiments.ball_track_labeler.contract) |
| `cv/experiments/ball_track_labeler/contract.py` | product closure (import; first parent: cv.experiments.ball_track_labeler) |
| `cv/experiments/ball_track_labeler/test_contract.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/__init__.py` | product closure (import; first parent: cv.experiments.connected_shooting.admissible_net_response) |
| `cv/experiments/connected_shooting/admissible_net_response.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/agent_attempt_prepare.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_single_flight_search) |
| `cv/experiments/connected_shooting/agent_single_flight_search.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/agent_whole_point_search.py` | product closure (import; first parent: cv.experiments.connected_shooting.auto_packet) |
| `cv/experiments/connected_shooting/athlete_priors.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_single_flight_search) |
| `cv/experiments/connected_shooting/auto_packet.py` | product closure (import; first parent: cv.pipeline.event_cascade_propose) |
| `cv/experiments/connected_shooting/block_flight_repair_probe.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_interval_block_fit) |
| `cv/experiments/connected_shooting/camera_geometry.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/candidate_attempts.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/cohort.py` | product closure (import; first parent: cv.experiments.connected_shooting.grass_bounce_profile) |
| `cv/experiments/connected_shooting/contact_geometry.py` | product closure (import; first parent: cv.experiments.connected_shooting.reach_constraints) |
| `cv/experiments/connected_shooting/contact_reach.py` | product closure (import; first parent: cv.experiments.connected_shooting.contact_geometry) |
| `cv/experiments/connected_shooting/depth_conditioned_seed.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/ending_witnesses.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/event_constraints.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/event_recovery.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/exposure_support.py` | product closure (import; first parent: cv.experiments.connected_shooting.pose_contact_feasibility) |
| `cv/experiments/connected_shooting/fast_flight.py` | product closure (import; first parent: cv.experiments.connected_shooting.ground_directed_seed) |
| `cv/experiments/connected_shooting/feasible_difference.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_terminal_ground_coupling) |
| `cv/experiments/connected_shooting/fit_ground_witness_center.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/flight_cache.py` | product closure (import; first parent: cv.experiments.connected_shooting.block_flight_repair_probe) |
| `cv/experiments/connected_shooting/full_native_continuation.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_context_census) |
| `cv/experiments/connected_shooting/grass_bounce_profile.py` | product closure (string target; first parent: physics.bounce_reference (string target)) |
| `cv/experiments/connected_shooting/ground_contact.py` | product closure (import; first parent: cv.experiments.connected_shooting.measured_dynamics) |
| `cv/experiments/connected_shooting/ground_directed_seed.py` | product closure (import; first parent: cv.experiments.connected_shooting.initialization) |
| `cv/experiments/connected_shooting/ground_root.py` | product closure (import; first parent: cv.experiments.connected_shooting.measured_dynamics) |
| `cv/experiments/connected_shooting/human_audit.py` | product closure (import; first parent: cv.experiments.connected_shooting.contact_geometry) |
| `cv/experiments/connected_shooting/human_replay.py` | product closure (import; first parent: cv.experiments.connected_shooting.human_audit) |
| `cv/experiments/connected_shooting/initialization.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/interior_contact_epochs.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/joint_toss_residual.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_player_prior_fit) |
| `cv/experiments/connected_shooting/labeled_attempt_sweep.py` | product closure (import; first parent: cv.experiments.connected_shooting.cohort) |
| `cv/experiments/connected_shooting/labeled_common_source.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/labeled_contact_association.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_common_source) |
| `cv/experiments/connected_shooting/labeled_context_census.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_context_witness_scope) |
| `cv/experiments/connected_shooting/labeled_context_witness_scope.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_interval_block_fit) |
| `cv/experiments/connected_shooting/labeled_event_occurrence.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/experiments/connected_shooting/labeled_free_toss_support.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_isolated_serve) |
| `cv/experiments/connected_shooting/labeled_interior_ground_response.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_interior_normal) |
| `cv/experiments/connected_shooting/labeled_interior_normal.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_interior_sweep) |
| `cv/experiments/connected_shooting/labeled_interior_sweep.py` | product closure (import; first parent: cv.pipeline.s6_shared_refinement) |
| `cv/experiments/connected_shooting/labeled_interval_block_fit.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_interior_normal) |
| `cv/experiments/connected_shooting/labeled_isolated_serve.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_free_toss_support) |
| `cv/experiments/connected_shooting/labeled_isolated_serve_fit.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_isolated_serve) |
| `cv/experiments/connected_shooting/labeled_missing_first_player.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_contact_association) |
| `cv/experiments/connected_shooting/labeled_missing_launch_association.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_contact_association) |
| `cv/experiments/connected_shooting/labeled_net_context_fit.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_net_recipe) |
| `cv/experiments/connected_shooting/labeled_net_epoch_chart.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/labeled_net_epoch_fit.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_net_context_fit) |
| `cv/experiments/connected_shooting/labeled_net_free_response.py` | product closure (import; first parent: cv.experiments.connected_shooting.admissible_net_response) |
| `cv/experiments/connected_shooting/labeled_net_ground_guidance.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_net_context_fit) |
| `cv/experiments/connected_shooting/labeled_net_ground_seed.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_net_context_fit) |
| `cv/experiments/connected_shooting/labeled_net_height_chart.py` | product closure (import; first parent: cv.experiments.connected_shooting.initialization) |
| `cv/experiments/connected_shooting/labeled_net_normal_response.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_net_epoch_fit) |
| `cv/experiments/connected_shooting/labeled_net_ray_seed.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_net_recipe) |
| `cv/experiments/connected_shooting/labeled_net_recipe.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_net_recipe (string target)) |
| `cv/experiments/connected_shooting/labeled_net_seed_family.py` | product closure (import; first parent: cv.pipeline.s6_shared_refinement) |
| `cv/experiments/connected_shooting/labeled_net_signed_response.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_net_free_response) |
| `cv/experiments/connected_shooting/labeled_net_spin_response.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_net_context_fit) |
| `cv/experiments/connected_shooting/labeled_nonnet_terminal_fit.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_prefix_joint_impact) |
| `cv/experiments/connected_shooting/labeled_nonnet_triage.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_nonnet_terminal_fit) |
| `cv/experiments/connected_shooting/labeled_passive_tape.py` | product closure (import; first parent: cv.experiments.connected_shooting.admissible_net_response) |
| `cv/experiments/connected_shooting/labeled_player_prior_fit.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_serve_contact_bounce_fit) |
| `cv/experiments/connected_shooting/labeled_player_serve_prior.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_player_prior_fit) |
| `cv/experiments/connected_shooting/labeled_prefix_boundary.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_interior_normal) |
| `cv/experiments/connected_shooting/labeled_prefix_joint_impact.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_interior_normal) |
| `cv/experiments/connected_shooting/labeled_prefix_later_net.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_prefix_joint_impact) |
| `cv/experiments/connected_shooting/labeled_prefix_net_chart.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_prefix_joint_impact) |
| `cv/experiments/connected_shooting/labeled_preparation_net_followup.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_interior_normal) |
| `cv/experiments/connected_shooting/labeled_preparation_net_recovery.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_missing_first_player) |
| `cv/experiments/connected_shooting/labeled_preparation_net_witness.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_preparation_net_followup) |
| `cv/experiments/connected_shooting/labeled_preparation_physical_seed.py` | product closure (import; first parent: cv.experiments.connected_shooting.observation_net_seed) |
| `cv/experiments/connected_shooting/labeled_preparation_recovery_association.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_missing_first_player) |
| `cv/experiments/connected_shooting/labeled_preparation_recovery_inputs.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_preparation_recovery_association) |
| `cv/experiments/connected_shooting/labeled_preparation_recovery_probe.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_preparation_net_recovery) |
| `cv/experiments/connected_shooting/labeled_preparation_recovery_review.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_nonnet_triage) |
| `cv/experiments/connected_shooting/labeled_serve_contact_bounce_fit.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_isolated_serve_fit) |
| `cv/experiments/connected_shooting/labeled_serve_recipe.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_common_source) |
| `cv/experiments/connected_shooting/labeled_terminal_ground_coupling.py` | product closure (import; first parent: cv.pipeline.s6_shared_refinement) |
| `cv/experiments/connected_shooting/labeled_terminal_ground_response.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_terminal_ground_coupling) |
| `cv/experiments/connected_shooting/labeled_terminal_impact.py` | product closure (import; first parent: cv.pipeline.s6_shared_refinement) |
| `cv/experiments/connected_shooting/labeled_terminal_net_coupling.py` | product closure (import; first parent: cv.pipeline.s6_shared_refinement) |
| `cv/experiments/connected_shooting/labeled_terminal_net_tail.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/labeled_terminal_tail_response.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_net_recipe) |
| `cv/experiments/connected_shooting/labeled_toss_admission.py` | product closure (import; first parent: cv.pipeline.s6_shared_refinement) |
| `cv/experiments/connected_shooting/labeled_toss_front.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_isolated_serve_fit) |
| `cv/experiments/connected_shooting/labeled_toss_player_anchor.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_isolated_serve_fit) |
| `cv/experiments/connected_shooting/labeled_toss_velocity_initializer.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_prefix_joint_impact) |
| `cv/experiments/connected_shooting/labeled_unsupported_contact_camera.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_contact_association) |
| `cv/experiments/connected_shooting/leading_edge_capacity.py` | product closure (import; first parent: cv.experiments.connected_shooting.real_exposure_replay) |
| `cv/experiments/connected_shooting/local_flight_repair_probe.py` | product closure (import; first parent: cv.experiments.connected_shooting.block_flight_repair_probe) |
| `cv/experiments/connected_shooting/measured_dynamics.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/model.py` | product closure (import; first parent: cv.experiments.connected_shooting.admissible_net_response) |
| `cv/experiments/connected_shooting/native_seed_check_pixels.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/nested_rebound.py` | product closure (import; first parent: cv.experiments.connected_shooting.model) |
| `cv/experiments/connected_shooting/net_collision.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/net_constraints.py` | product closure (import; first parent: cv.experiments.connected_shooting.event_constraints) |
| `cv/experiments/connected_shooting/net_cord_evidence.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/observation_net_seed.py` | product closure (import; first parent: cv.experiments.connected_shooting.admissible_net_response) |
| `cv/experiments/connected_shooting/observation_operator.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/observation_partition.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/observation_scope.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/observed_horizon_tail.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/oracle_benchmark.py` | product closure (import; first parent: cv.experiments.connected_shooting.leading_edge_capacity) |
| `cv/experiments/connected_shooting/passive_bounce.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/per_flight_acceptance.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_serve_recipe) |
| `cv/experiments/connected_shooting/per_flight_pictures.py` | product closure (import; first parent: cv.experiments.connected_shooting.local_flight_repair_probe) |
| `cv/experiments/connected_shooting/per_flight_rescore.py` | product closure (import; first parent: cv.experiments.connected_shooting.full_native_continuation) |
| `cv/experiments/connected_shooting/physical_audit.py` | product closure (import; first parent: cv.experiments.connected_shooting.human_audit) |
| `cv/experiments/connected_shooting/physical_compatibility.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_prefix_boundary) |
| `cv/experiments/connected_shooting/player_position.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/player_state_fallback.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/pose_contact_feasibility.py` | product closure (import; first parent: cv.experiments.connected_shooting.serve_region_capacity) |
| `cv/experiments/connected_shooting/postbounce_initialization.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/prefix_following_ground_timing.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_prefix_joint_impact) |
| `cv/experiments/connected_shooting/prefix_pixel_loss.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_prefix_joint_impact) |
| `cv/experiments/connected_shooting/reach_constraints.py` | product closure (import; first parent: cv.experiments.connected_shooting.model) |
| `cv/experiments/connected_shooting/real_bidirectional_search.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_single_flight_search) |
| `cv/experiments/connected_shooting/real_exposure_replay.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_single_flight_search) |
| `cv/experiments/connected_shooting/regime_recovery.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_prefix_joint_impact) |
| `cv/experiments/connected_shooting/search_budget.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/search_reporting.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/serve_block_repair_probe.py` | product closure (import; first parent: cv.experiments.connected_shooting.serve_timing_profile) |
| `cv/experiments/connected_shooting/serve_reach_cylinder.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/serve_region_capacity.py` | product closure (import; first parent: cv.experiments.connected_shooting.leading_edge_capacity) |
| `cv/experiments/connected_shooting/serve_side.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_single_flight_search) |
| `cv/experiments/connected_shooting/serve_speed_witness.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/serve_timing_profile.py` | product closure (import; first parent: cv.experiments.connected_shooting.full_native_continuation) |
| `cv/experiments/connected_shooting/source_flight_coverage.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/experiments/connected_shooting/streak_axis_capacity.py` | product closure (import; first parent: cv.experiments.connected_shooting.real_exposure_replay) |
| `cv/experiments/connected_shooting/swept_exposure.py` | product closure (import; first parent: cv.experiments.connected_shooting.grass_bounce_profile) |
| `cv/experiments/connected_shooting/terminal_completion.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_context_census) |
| `cv/experiments/connected_shooting/terminal_exposure_context.py` | product closure (import; first parent: cv.experiments.connected_shooting.leading_edge_capacity) |
| `cv/experiments/connected_shooting/terminal_feasibility.py` | product closure (import; first parent: cv.experiments.connected_shooting.leading_edge_capacity) |
| `cv/experiments/connected_shooting/test_admissible_net_response.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_agent_attempt_prepare.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_agent_whole_point_search.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_anchor_net_initialization.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_athlete_evidence_policy.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_athlete_priors.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_athlete_root_reach_loss.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_auto_packet.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_automatic_toss_semantics.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_block_flight_repair_probe.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_candidate_attempts.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_cohort.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_contact_geometry.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_contact_reach.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_depth_conditioned_seed.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_ending_witnesses.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_event_constraints.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_event_recovery.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_fast_flight.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_final_bounce_anchor.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_fit_ground_witness_center.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_full_native_continuation.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_grass_bounce_profile.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_ground_contact.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_ground_epoch_launch.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_human_audit.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_initialization.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_joint_toss_residual.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_attempt_sweep.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_cincy_adjacent_contact.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_contact_players.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_context_census.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_free_toss_support.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_interior_ground_response.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_missing_first_player.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_missing_launch_association.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_epoch_chart.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_epoch_inclination.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_free_response.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_ground_guidance.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_ground_seed.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_height_chart.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_mesh_height.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_normal_response.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_ray_seed.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_recipe.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_seed_family.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_signed_response.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_spin_integration.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_net_spin_response.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_nonnet_terminal_fit.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_player_serve_prior.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_preparation_net_followup.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_preparation_net_recovery.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_preparation_net_witness.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_preparation_physical_seed.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_preparation_recovery_association.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_preparation_recovery_inputs.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_preparation_recovery_probe.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_serve_contact_bounce_fit.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_serve_recipe.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_toss_admission.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_toss_front.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_toss_player_anchor.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_labeled_toss_velocity_initializer.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_local_flight_repair_probe.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_major_boundary_restoration.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_measured_dynamics.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_model.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_nested_rebound.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_net_collision.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_net_collision_physical_eligibility.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_net_cord_evidence.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_net_cord_response_receipt.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_observation_fallback.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_observed_horizon_event_boundary.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_oracle_benchmark.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_owner_approved_gate_receipt.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_per_flight_acceptance.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_player_state_fallback.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_postbounce_initialization.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_prefix_regime_retry.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_real_bidirectional_search.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_real_exposure_replay.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_restart_safety.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_search_budget.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_search_incumbent.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_serve_block_repair_probe.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_serve_reach_cylinder.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_serve_region_capacity.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_serve_side.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_serve_speed_witness.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_serve_timing_profile.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_shared_contact_states.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_source_flight_coverage.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_terminal_completion.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/test_terminal_seed_start.py` | unit test whose repository imports are all included |
| `cv/experiments/connected_shooting/toss_witness.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/active_play.py` | product closure (import; first parent: cv.pipeline.active_play_gate) |
| `cv/pipeline/active_play_gate.py` | product closure (string target; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/anchor_first_fit.py` | product closure (import; first parent: cv.pipeline.reconstruct_3d) |
| `cv/pipeline/anthropometric_contacts.py` | product closure (import; first parent: cv.experiments.connected_shooting.initialization) |
| `cv/pipeline/artifact_cache.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/artifact_lineage.json` | artifact lineage contract read by the pipeline |
| `cv/pipeline/attempt_observation_scope.py` | product closure (import; first parent: cv.pipeline.point_ledger) |
| `cv/pipeline/audio_evidence.py` | product closure (string target; first parent: cv.pipeline.run_contacts_pipeline (string target)) |
| `cv/pipeline/audio_features.py` | product closure (import; first parent: cv.pipeline.audio_evidence) |
| `cv/pipeline/automatic_ball_track.py` | product closure (import; first parent: cv.experiments.connected_shooting.auto_packet) |
| `cv/pipeline/ball.py` | product closure (import; first parent: cv.pipeline.ball_far2x (sys.path bare import)) |
| `cv/pipeline/ball_anchor_extension.py` | product closure (string target; first parent: cv.pipeline.tracking_composition (string target)) |
| `cv/pipeline/ball_events.py` | product closure (import; first parent: cv.pipeline.event_proposals) |
| `cv/pipeline/ball_far2x.py` | product closure (import; first parent: cv.pipeline.ball_neural) |
| `cv/pipeline/ball_local_refine.py` | product closure (import; first parent: cv.pipeline.ball_local_refine_batched) |
| `cv/pipeline/ball_local_refine_batched.py` | product closure (string target; first parent: cv.pipeline.tracking_composition (string target)) |
| `cv/pipeline/ball_motion_tracker.py` | product closure (import; first parent: cv.pipeline.ball_anchor_extension) |
| `cv/pipeline/ball_neural.py` | product closure (import; first parent: cv.pipeline.ball_far2x (sys.path bare import)) |
| `cv/pipeline/ball_neural_batched.py` | product closure (string target; first parent: cv.pipeline.tracking_composition (string target)) |
| `cv/pipeline/ball_ownership.py` | product closure (import; first parent: cv.pipeline.ball_anchor_extension) |
| `cv/pipeline/ball_track_consensus.py` | product closure (import; first parent: cv.pipeline.ball_track_detour_repair) |
| `cv/pipeline/ball_track_detour_repair.py` | product closure (string target; first parent: cv.pipeline.tracking_composition (string target)) |
| `cv/pipeline/ball_track_hypotheses.py` | product closure (import; first parent: cv.pipeline.tracking_point_gate (sys.path bare import)) |
| `cv/pipeline/ball_track_local_augment.py` | product closure (import; first parent: cv.pipeline.ball_track_detour_repair) |
| `cv/pipeline/bounce_detect.py` | product closure (import; first parent: cv.pipeline.active_play_gate) |
| `cv/pipeline/broadcast_runner.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/broadcast_source.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/cadence_normalize.py` | product closure (import; first parent: cv.pipeline.broadcast_source) |
| `cv/pipeline/camera.py` | product closure (string target; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/camera_P_per_frame.py` | product closure (import; first parent: cv.pipeline.camera_artifacts) |
| `cv/pipeline/camera_artifacts.py` | product closure (import; first parent: cv.experiments.connected_shooting.auto_packet) |
| `cv/pipeline/camera_bundle.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/pipeline/camera_cal.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/pipeline/camera_frame_support.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/camera_metric_refine.py` | product closure (string target; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/camera_project.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/pipeline/camera_scope.py` | product closure (import; first parent: cv.pipeline.reconstruction) |
| `cv/pipeline/canonical_runner.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/compose_point_validity_gate.py` | product closure (string target; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/contact_frame_refiner.py` | product closure (import; first parent: cv.pipeline.contact_recall) |
| `cv/pipeline/contact_recall.py` | product closure (import; first parent: cv.pipeline.event_proposals) |
| `cv/pipeline/contact_striker.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/contacts_audio.py` | product closure (import; first parent: cv.pipeline.event_contact_witness) |
| `cv/pipeline/contacts_v2.py` | product closure (import; first parent: cv.pipeline.contacts_audio (sys.path bare import)) |
| `cv/pipeline/court.py` | product closure (import; first parent: cv.pipeline.court_far_baseline_refinement) |
| `cv/pipeline/court_anchor_camera.py` | product closure (import; first parent: cv.pipeline.active_play) |
| `cv/pipeline/court_far_baseline_refinement.py` | product closure (import; first parent: cv.pipeline.court_topology_runner) |
| `cv/pipeline/court_geometry_gate.py` | product closure (import; first parent: cv.pipeline.court) |
| `cv/pipeline/court_geometry_point_gate.py` | product closure (string target; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/court_near_baseline_refinement.py` | product closure (import; first parent: cv.pipeline.court_paint_refinement) |
| `cv/pipeline/court_paint_refinement.py` | product closure (import; first parent: cv.pipeline.s6_component_scope) |
| `cv/pipeline/court_topology.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/pipeline/court_topology_frame_track.py` | product closure (import; first parent: cv.pipeline.court_topology_frame_track (string target)) |
| `cv/pipeline/court_topology_runner.py` | product closure (import; first parent: cv.pipeline.camera_artifacts) |
| `cv/pipeline/default_gates.json` | default flight/physics gates read by the pipeline |
| `cv/pipeline/evaluation_scope.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/event_cascade.py` | product closure (import; first parent: cv.pipeline.event_cascade_models) |
| `cv/pipeline/event_cascade_frozen/__init__.py` | product closure (import; first parent: cv.pipeline.event_cascade_frozen.render_evidence) |
| `cv/pipeline/event_cascade_frozen/fairlib.py` | product closure (import; first parent: cv.pipeline.event_cascade_frozen.render_evidence) |
| `cv/pipeline/event_cascade_frozen/guided_procedure.py` | product closure (import; first parent: cv.pipeline.event_cascade_frozen.run_label) |
| `cv/pipeline/event_cascade_frozen/or_call.py` | product closure (import; first parent: cv.pipeline.event_cascade_frozen.run_label) |
| `cv/pipeline/event_cascade_frozen/render_evidence.py` | product closure (import; first parent: cv.pipeline.event_cascade_models) |
| `cv/pipeline/event_cascade_frozen/run_label.py` | product closure (import; first parent: cv.pipeline.event_cascade_models) |
| `cv/pipeline/event_cascade_frozen/trace.py` | product closure (import; first parent: cv.pipeline.event_cascade_frozen.render_evidence) |
| `cv/pipeline/event_cascade_frozen/trace_from_track.py` | product closure (import; first parent: cv.pipeline.event_cascade_frozen.render_evidence) |
| `cv/pipeline/event_cascade_models.py` | product closure (import; first parent: cv.pipeline.event_cascade_propose) |
| `cv/pipeline/event_cascade_propose.py` | product closure (import; first parent: cv.pipeline.event_cascade_runner) |
| `cv/pipeline/event_cascade_rebind.py` | product closure (import; first parent: cv.pipeline.event_cascade_propose) |
| `cv/pipeline/event_cascade_runner.py` | product closure (import; first parent: cv.pipeline.event_cascade_runner (string target)) |
| `cv/pipeline/event_contact_witness.py` | product closure (import; first parent: cv.pipeline.s6_contact_prefix_scope) |
| `cv/pipeline/event_crop_translation.py` | product closure (import; first parent: cv.pipeline.event_crop_translation (string target)) |
| `cv/pipeline/event_crops.py` | product closure (import; first parent: cv.pipeline.event_crop_translation) |
| `cv/pipeline/event_decoder.py` | product closure (import; first parent: cv.pipeline.event_inference) |
| `cv/pipeline/event_grammar_decoder.py` | product closure (import; first parent: cv.pipeline.event_ground_evidence) |
| `cv/pipeline/event_ground_evidence.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/event_impulse_support.py` | product closure (import; first parent: cv.pipeline.event_ground_evidence) |
| `cv/pipeline/event_inference.py` | product closure (import; first parent: cv.pipeline.point_grammar) |
| `cv/pipeline/event_model.py` | product closure (import; first parent: cv.pipeline.event_inference) |
| `cv/pipeline/event_model_admission.py` | product closure (import; first parent: cv.pipeline.event_video_model) |
| `cv/pipeline/event_model_v2.py` | product closure (import; first parent: cv.pipeline.event_model_v3) |
| `cv/pipeline/event_model_v2_features.py` | product closure (import; first parent: cv.pipeline.camera_frame_support) |
| `cv/pipeline/event_model_v3.py` | product closure (string target; first parent: cv.pipeline.canonical_runner (string target)) |
| `cv/pipeline/event_native_pictures.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/event_net_evidence.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/event_paths.py` | product closure (import; first parent: cv.experiments.connected_shooting.auto_packet) |
| `cv/pipeline/event_proposals.py` | product closure (import; first parent: cv.pipeline.event_crops) |
| `cv/pipeline/event_refine.py` | product closure (import; first parent: cv.pipeline.ball_track_detour_repair) |
| `cv/pipeline/event_sequence_graph.py` | product closure (import; first parent: cv.pipeline.event_decoder) |
| `cv/pipeline/event_time_distribution.py` | product closure (import; first parent: cv.pipeline.event_grammar_decoder) |
| `cv/pipeline/event_time_neighborhood.py` | product closure (import; first parent: cv.pipeline.event_grammar_decoder) |
| `cv/pipeline/event_topology.py` | product closure (import; first parent: cv.pipeline.reconstruction) |
| `cv/pipeline/event_video_model.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/event_wing_streak.py` | product closure (import; first parent: cv.pipeline.event_impulse_support) |
| `cv/pipeline/flight_anchors.py` | product closure (import; first parent: cv.pipeline.anchor_first_fit) |
| `cv/pipeline/flight_ledger.py` | product closure (import; first parent: cv.pipeline.reconstruction) |
| `cv/pipeline/frame_cadence.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/frame_identity.py` | product closure (import; first parent: cv.pipeline.ball) |
| `cv/pipeline/guide_sequence_association.py` | product closure (import; first parent: cv.pipeline.ball_motion_tracker) |
| `cv/pipeline/implementations.json` | implementation registry read by point_context |
| `cv/pipeline/live_shot_camera.py` | product closure (import; first parent: cv.pipeline.event_crops) |
| `cv/pipeline/native_actor_continuity.py` | product closure (import; first parent: cv.pipeline.play_camera_leakage) |
| `cv/pipeline/native_shot_continuity.py` | product closure (import; first parent: cv.pipeline.shot_segments) |
| `cv/pipeline/net_cord_response.py` | product closure (import; first parent: cv.experiments.connected_shooting.admissible_net_response) |
| `cv/pipeline/paths.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/pipeline/physics_events.py` | product closure (import; first parent: cv.pipeline.rich_ball_physics (sys.path bare import)) |
| `cv/pipeline/physics_interpretation.py` | product closure (import; first parent: cv.pipeline.reconstruction) |
| `cv/pipeline/physics_knot_solver.py` | product closure (import; first parent: cv.experiments.connected_shooting.initialization) |
| `cv/pipeline/pipeline_evidence.py` | product closure (import; first parent: cv.pipeline.reconstruction) |
| `cv/pipeline/play_camera_leakage.py` | product closure (import; first parent: cv.pipeline.active_play_gate) |
| `cv/pipeline/play_phase.py` | product closure (import; first parent: cv.pipeline.active_play) |
| `cv/pipeline/player_biometrics.json` | public player heights (ATP/WTA profile URLs) |
| `cv/pipeline/player_court_v2.py` | product closure (import; first parent: cv.pipeline.player_side_association) |
| `cv/pipeline/player_identity.py` | product closure (import; first parent: cv.pipeline.player_side_association) |
| `cv/pipeline/player_motion_physical.py` | product closure (string target; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/player_side_association.py` | product closure (import; first parent: cv.pipeline.canonical_runner (string target)) |
| `cv/pipeline/player_tracker.py` | product closure (import; first parent: cv.pipeline.native_actor_continuity) |
| `cv/pipeline/players.py` | product closure (import; first parent: cv.pipeline.run_contacts_pipeline (string target)) |
| `cv/pipeline/point_context.py` | product closure (import; first parent: cv.experiments.connected_shooting.ending_witnesses) |
| `cv/pipeline/point_end_tag.py` | product closure (import; first parent: cv.pipeline.product_runner) |
| `cv/pipeline/point_grammar.py` | product closure (import; first parent: cv.experiments.connected_shooting.auto_packet) |
| `cv/pipeline/point_ledger.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/point_validity.py` | product closure (import; first parent: cv.pipeline.camera_frame_support) |
| `cv/pipeline/pose.py` | product closure (import; first parent: cv.pipeline.contact_striker) |
| `cv/pipeline/pose_crop_infer.py` | product closure (import; first parent: cv.pipeline.pose_player_crop (sys.path bare import)) |
| `cv/pipeline/pose_lift.py` | product closure (import; first parent: cv.viz.export_connected_3d) |
| `cv/pipeline/pose_player_crop.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/product_runner.py` | product closure (import; first parent: cv.pipeline.product_runner (string target)) |
| `cv/pipeline/product_s6_policy.json` | production S6 composed policy read by product_runner |
| `cv/pipeline/provenance.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/pipeline/reconstruct_3d.py` | product closure (string target; first parent: cv.validation.s6_cohort_v2_eval (string target)) |
| `cv/pipeline/reconstruction.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/resolution.py` | product closure (import; first parent: cv.experiments.connected_shooting.auto_packet) |
| `cv/pipeline/rich_ball_physics.py` | product closure (import; first parent: cv.experiments.connected_shooting.ground_directed_seed) |
| `cv/pipeline/run_contacts_pipeline.py` | product closure (string target; first parent: cv.pipeline.canonical_runner (string target)) |
| `cv/pipeline/run_manifest.py` | product closure (import; first parent: cv.pipeline.anthropometric_contacts (sys.path bare import)) |
| `cv/pipeline/s6_attempt_execution.py` | product closure (import; first parent: cv.pipeline.s6_broadcast_backend) |
| `cv/pipeline/s6_automatic_observations.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/s6_ball_jump_rejection.py` | product closure (import; first parent: cv.pipeline.s6_automatic_observations) |
| `cv/pipeline/s6_ball_streak_centre.py` | product closure (import; first parent: cv.pipeline.s6_automatic_observations) |
| `cv/pipeline/s6_broadcast_backend.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/s6_component_orchestration.py` | product closure (import; first parent: cv.pipeline.s6_attempt_execution) |
| `cv/pipeline/s6_component_scope.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_contact_components.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_contact_composition.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_contact_prefix_runtime.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_contact_prefix_scope.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_contact_timing.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_event_operating_point.py` | product closure (import; first parent: cv.pipeline.s6_automatic_observations) |
| `cv/pipeline/s6_first_contact_role.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_first_flight_scope.py` | product closure (import; first parent: cv.experiments.connected_shooting.observation_scope) |
| `cv/pipeline/s6_independent_event_proposals.py` | product closure (import; first parent: cv.pipeline.s6_labeled_stage) |
| `cv/pipeline/s6_input_origin.py` | product closure (import; first parent: cv.pipeline.event_cascade_rebind) |
| `cv/pipeline/s6_labeled_stage.py` | product closure (import; first parent: cv.experiments.connected_shooting.auto_packet) |
| `cv/pipeline/s6_optional_bounce_scope.py` | product closure (import; first parent: cv.experiments.connected_shooting.observation_scope) |
| `cv/pipeline/s6_optional_bounces.py` | product closure (import; first parent: cv.pipeline.s6_labeled_stage) |
| `cv/pipeline/s6_optional_contacts.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_optional_event_union.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_owner_gate_loosening.py` | product closure (import; first parent: cv.pipeline.s6_labeled_stage) |
| `cv/pipeline/s6_player_camera.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_common_source) |
| `cv/pipeline/s6_preparation_policy.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_producer_source_recovery.py` | product closure (import; first parent: cv.pipeline.s6_automatic_observations) |
| `cv/pipeline/s6_rally_origin_cue.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_refinement_input_context.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_common_source) |
| `cv/pipeline/s6_shared_refinement.py` | product closure (import; first parent: cv.pipeline.s6_component_orchestration) |
| `cv/pipeline/s6_terminal_context.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_terminal_net_membership.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/s6_witnessed_interior_contacts.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/score_grammar.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/score_vlm.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/segment_boundaries.py` | product closure (import; first parent: cv.pipeline.service_attempt_scope) |
| `cv/pipeline/serve_audio.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/serve_contact_detect.py` | product closure (import; first parent: cv.pipeline.ball_events (sys.path bare import)) |
| `cv/pipeline/serve_detector.py` | product closure (string target; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/serve_hints.py` | product closure (import; first parent: cv.pipeline.ball_events (sys.path bare import)) |
| `cv/pipeline/serve_location_prior.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/pipeline/serve_speed_graphic.py` | product closure (import; first parent: cv.pipeline.point_context) |
| `cv/pipeline/service_attempt_scope.py` | product closure (import; first parent: cv.pipeline.s6_automatic_observations) |
| `cv/pipeline/shot_boundaries.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/shot_segments.py` | product closure (import; first parent: cv.pipeline.active_play) |
| `cv/pipeline/shots_from_video.py` | product closure (import; first parent: cv.pipeline.ball (sys.path bare import)) |
| `cv/pipeline/source_timebase.py` | product closure (import; first parent: cv.pipeline.broadcast_source) |
| `cv/pipeline/subframe_timing.py` | product closure (import; first parent: cv.pipeline.flight_anchors) |
| `cv/pipeline/terminal_completion.py` | product closure (import; first parent: cv.pipeline.reconstruction) |
| `cv/pipeline/test_active_play.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_active_play_diagnostic_origin.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_active_play_gate.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_anchor_first_fit.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_anthropometric_contacts.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_artifact_cache.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_attempt_observation_scope.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_automatic_ball_track.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_anchor_extension.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_events_kinematics.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_local_refine.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_local_refine_batched.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_motion_tracker.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_neural.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_neural_batched.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_ownership.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_track_consensus.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_track_detour_repair.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_track_hypotheses.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_ball_track_local_augment.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_bounce_detect.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_broadcast_source.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_camera.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_camera_P_per_frame.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_camera_artifacts.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_camera_bundle.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_camera_cal.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_camera_frame_support.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_camera_metric_anchor.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_camera_metric_refine.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_camera_registration_mask.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_camera_scope.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_canonical_runner.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_contact_frame_refiner.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_contact_recall.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_contact_striker.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_contacts_audio.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_contacts_v2.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_coordinate_columns.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_court_anchor_camera.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_court_far_baseline_refinement.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_court_geometry_gate.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_court_joint_camera.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_court_paint_refinement.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_court_proposal_fallback.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_court_topology.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_court_topology_frame_track.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_court_topology_runner.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_evaluation_scope.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_cascade.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_cascade_frozen.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_cascade_propose.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_contact_witness.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_crop_translation.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_crops.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_decoder.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_grammar_decoder.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_impulse_support.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_inference.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_model.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_model_v2.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_model_v2_features.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_model_v3.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_model_v3_hypotheses.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_paths.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_proposal_modalities.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_proposals.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_refine.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_sequence_graph.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_time_distribution.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_topology.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_event_wing_streak.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_flight_anchors.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_flight_ledger.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_frame_cadence.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_frame_identity.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_guide_sequence_association.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_implementation_registry.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_independent_contact_pose.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_native_actor_continuity.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_native_extraction_epochs.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_native_image_support.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_native_shot_continuity.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_net_cord_artifact.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_paths.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_physics_events.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_physics_interpretation.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_physics_knot_solver_audio.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_physics_knot_solver_deadball.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_physics_knot_solver_net.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_physics_knot_solver_noise.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_physics_knot_solver_terminal.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_pipeline_evidence.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_play_camera_leakage.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_play_phase.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_player_court_v2.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_player_identity.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_player_motion_physical.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_player_side_association.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_player_tracker.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_players.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_players_point_map.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_point_context.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_point_end_tag.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_point_grammar.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_point_ledger.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_point_validity.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_pose.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_pose_crop_infer.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_pose_lift.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_pose_player_crop.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_product_runner.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_provenance.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_reconstruct_3d.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_reconstruction.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_reconstruction_event_scope.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_resolution.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_resolution_contract_v2.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_rich_ball_physics.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_run_contacts_pipeline.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_automatic_observations.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_ball_jump_rejection.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_ball_streak_centre.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_broadcast_backend.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_contact_composition.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_contact_prefix_scope.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_event_operating_point.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_final_contact_scope.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_first_flight_scope.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_independent_event_proposals.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_optional_contacts.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_optional_rebound_ownership.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_owner_gate_loosening.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_s6_witnessed_interior_contacts.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_score_grammar.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_score_vlm.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_segment_boundaries.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_serve_contact_detect.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_serve_detector.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_serve_location_prior.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_serve_speed_graphic.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_service_attempt_scope.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_shot_boundaries.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_shot_segments.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_source_timebase.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_subframe_timing.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_terminal_completion.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_torso_lock.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_track_heal.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_tracking_composition_imports.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_tracking_point_gate.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_trajectory_contract.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_upstream_gate_release.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_view_classifier.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_visual_play_coverage.py` | unit test whose repository imports are all included |
| `cv/pipeline/test_vlm_policy.py` | unit test whose repository imports are all included |
| `cv/pipeline/testdata/optional_rebound_ownership_source02.json` | synthetic fixtures for included unit tests |
| `cv/pipeline/torso_lock.py` | product closure (import; first parent: cv.pipeline.ball_motion_tracker) |
| `cv/pipeline/track_artifact.py` | product closure (import; first parent: cv.pipeline.ball_anchor_extension) |
| `cv/pipeline/track_heal.py` | product closure (import; first parent: cv.pipeline.reconstruction) |
| `cv/pipeline/tracking_composition.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/tracking_point_gate.py` | product closure (string target; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/trajectory_contract.py` | product closure (import; first parent: cv.experiments.connected_shooting.per_flight_acceptance) |
| `cv/pipeline/view_classifier.py` | product closure (import; first parent: cv.pipeline.broadcast_runner (string target)) |
| `cv/pipeline/vlm_models.json` | model registry for the optional scoreboard/cascade VLMs |
| `cv/pipeline/vlm_policy.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/pipeline/window_camera_inference.py` | product closure (import; first parent: cv.pipeline.broadcast_runner) |
| `cv/validation/ball_streak_reference.py` | product closure (import; first parent: cv.validation.s6_owner_inputs) |
| `cv/validation/current_standard_event_truth.py` | product closure (import; first parent: cv.validation.event_pixel_accuracy) |
| `cv/validation/event_pixel_accuracy.py` | product closure (import; first parent: cv.validation.s6_bench) |
| `cv/validation/flight_gate_audit.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/validation/ground_truth.py` | product closure (import; first parent: cv.pipeline.contacts_v2 (sys.path bare import)) |
| `cv/validation/hawkeye_bounce_model.py` | product closure (import; first parent: entry) |
| `cv/validation/hawkeye_flight_fit.py` | product closure (import; first parent: cv.validation.hawkeye_bounce_model) |
| `cv/validation/oracle_3d_ceiling.py` | product closure (import; first parent: cv.validation.s6_cohort_v2_eval) |
| `cv/validation/owner_spatial_audit.py` | product closure (import; first parent: cv.experiments.connected_shooting.oracle_benchmark) |
| `cv/validation/physics_validation.py` | product closure (import; first parent: cv.validation.s6_bench) |
| `cv/validation/player_truth_ledger.py` | product closure (import; first parent: cv.viz.export_connected_3d) |
| `cv/validation/playstyle_pattern.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_whole_point_search) |
| `cv/validation/run_labeled_s6.py` | product closure (string target; first parent: cv.pipeline.s6_labeled_stage (string target)) |
| `cv/validation/s6_bench.py` | product closure (import; first parent: cv.validation.s6_bench (string target)) |
| `cv/validation/s6_cohort_v2_eval.py` | product closure (import; first parent: cv.validation.event_pixel_accuracy) |
| `cv/validation/s6_owner_camera_transport.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/validation/s6_owner_court_audit.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/validation/s6_owner_ground_camera.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/validation/s6_owner_inputs.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_attempt_prepare) |
| `cv/validation/s6_point_bench.py` | product closure (import; first parent: cv.experiments.connected_shooting.oracle_benchmark) |
| `cv/validation/s6_sparse_owner_replay.py` | product closure (import; first parent: cv.experiments.connected_shooting.agent_single_flight_search) |
| `cv/validation/s6root_closed_loop.py` | product closure (import; first parent: cv.validation.physics_validation) |
| `cv/validation/s6root_common.py` | product closure (import; first parent: cv.validation.physics_validation) |
| `cv/validation/s6root_metric_acceptance.py` | product closure (import; first parent: cv.validation.flight_gate_audit) |
| `cv/validation/score_contact_strikers.py` | product closure (import; first parent: cv.validation.player_truth_ledger) |
| `cv/validation/score_cross_match_event_labels_v5.py` | product closure (import; first parent: cv.validation.event_pixel_accuracy) |
| `cv/validation/scoring.py` | product closure (import; first parent: cv.pipeline.shots_from_video (sys.path bare import)) |
| `cv/validation/test_ball_streak_reference.py` | unit test whose repository imports are all included |
| `cv/validation/test_event_pixel_accuracy.py` | unit test whose repository imports are all included |
| `cv/validation/test_flight_gate_audit.py` | unit test whose repository imports are all included |
| `cv/validation/test_hawkeye_bounce_model.py` | unit test whose repository imports are all included |
| `cv/validation/test_hawkeye_flight_fit.py` | unit test whose repository imports are all included |
| `cv/validation/test_oracle_3d_ceiling.py` | unit test whose repository imports are all included |
| `cv/validation/test_owner_spatial_audit.py` | unit test whose repository imports are all included |
| `cv/validation/test_player_truth_ledger.py` | unit test whose repository imports are all included |
| `cv/validation/test_playstyle_pattern.py` | unit test whose repository imports are all included |
| `cv/validation/test_run_labeled_s6.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6_bench.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6_cohort_v2_eval.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6_net_eligibility_policy.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6_owner_court_audit.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6_owner_ground_camera.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6_owner_inputs.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6_player_camera.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6_point_bench.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6_preparation_policy.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6_sparse_owner_replay.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6root_closed_loop.py` | unit test whose repository imports are all included |
| `cv/validation/test_s6root_metric_acceptance.py` | unit test whose repository imports are all included |
| `cv/validation/test_score_contact_strikers.py` | unit test whose repository imports are all included |
| `cv/validation/test_score_cross_match_event_labels_v5.py` | unit test whose repository imports are all included |
| `cv/validation/test_scoring.py` | unit test whose repository imports are all included |
| `cv/validation/test_untouched_tracking_pipeline_throughput.py` | unit test whose repository imports are all included |
| `cv/validation/test_wk3_s6_cohort.py` | unit test whose repository imports are all included |
| `cv/validation/wk3_s6_cohort.py` | product closure (import; first parent: cv.validation.oracle_3d_ceiling) |
| `cv/viz/export_automatic_s6.py` | product closure (import; first parent: cv.pipeline.product_runner) |
| `cv/viz/export_connected_3d.py` | product closure (import; first parent: cv.pipeline.product_runner) |
| `cv/viz/export_local_s6.py` | product closure (import; first parent: cv.pipeline.product_runner) |
| `cv/viz/export_point_3d.py` | product closure (import; first parent: cv.viz.export_connected_3d) |
| `cv/viz/portal_3d_src/build.py` | authored for the public release (overlay) |
| `cv/viz/portal_3d_src/index.html` | 3D flight viewer page built by cv.viz.portal_3d_src.build |
| `cv/viz/portal_3d_src/test_build.py` | authored for the public release (overlay) |
| `cv/viz/portal_3d_src/test_viewer_core.cjs` | 3D flight viewer page built by cv.viz.portal_3d_src.build |
| `cv/viz/portal_3d_src/vendor/OrbitControls.js` | vendored three.js r160 (MIT) for the 3D flight viewer |
| `cv/viz/portal_3d_src/vendor/THREE_LICENSE.txt` | vendored three.js r160 (MIT) for the 3D flight viewer |
| `cv/viz/portal_3d_src/vendor/VERSION.txt` | vendored three.js r160 (MIT) for the 3D flight viewer |
| `cv/viz/portal_3d_src/vendor/three.module.js` | vendored three.js r160 (MIT) for the 3D flight viewer |
| `cv/viz/portal_3d_src/viewer.js` | 3D flight viewer page built by cv.viz.portal_3d_src.build |
| `cv/viz/portal_3d_src/viewer_core.js` | 3D flight viewer page built by cv.viz.portal_3d_src.build |
| `cv/viz/test_contact_prefix_export.py` | unit test whose repository imports are all included |
| `cv/viz/test_export_connected_3d.py` | unit test whose repository imports are all included |
| `cv/viz/test_export_native_clock.py` | unit test whose repository imports are all included |
| `cv/viz/test_export_point_3d.py` | unit test whose repository imports are all included |
| `docs/DATA_SCHEMA.md` | authored for the public release (overlay) |
| `docs/EVALUATION.md` | authored for the public release (overlay) |
| `docs/MODELS.md` | authored for the public release (overlay) |
| `docs/PHYSICS.md` | authored for the public release (overlay) |
| `docs/samples/sample_rally1.gif` | README sample (render_samples.py; logo inpainted) |
| `docs/samples/sample_rally1.png` | README sample (render_samples.py; logo inpainted) |
| `docs/samples/sample_rally2.png` | README sample (render_samples.py; logo inpainted) |
| `evaluation/__init__.py` | authored for the public release (overlay) |
| `evaluation/origin_matcher.py` | authored for the public release (overlay) |
| `evaluation/test_origin_matcher.py` | authored for the public release (overlay) |
| `physics/REFERENCE.md` | physics reference notes |
| `physics/bounce_law_hawkeye_holdout.json` | production bounce law (fitted parameters only) |
| `physics/bounce_reference.py` | product closure (import; first parent: cv.experiments.connected_shooting.grass_bounce_profile (string target)) |
| `physics/flight.py` | product closure (import; first parent: cv.experiments.connected_shooting.fast_flight) |
| `physics/grass_surface_model.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_attempt_sweep) |
| `physics/impact.py` | product closure (import; first parent: cv.experiments.connected_shooting.grass_bounce_profile) |
| `physics/passive_bounce.py` | product closure (import; first parent: cv.experiments.connected_shooting.passive_bounce) |
| `physics/reference.py` | product closure (import; first parent: cv.validation.hawkeye_bounce_model) |
| `physics/surface_model.py` | product closure (import; first parent: cv.experiments.connected_shooting.ground_contact) |
| `physics/test_bounce_reference.py` | unit test whose repository imports are all included |
| `physics/test_flight.py` | unit test whose repository imports are all included |
| `physics/test_flight_spin_decay.py` | unit test whose repository imports are all included |
| `physics/test_grass_surface_model.py` | unit test whose repository imports are all included |
| `physics/test_ground_impact.py` | unit test whose repository imports are all included |
| `physics/test_impact_regime_surrogate.py` | unit test whose repository imports are all included |
| `physics/test_impact_vector.py` | unit test whose repository imports are all included |
| `physics/test_impact_vector_parity.py` | unit test whose repository imports are all included |
| `physics/test_passive_bounce.py` | unit test whose repository imports are all included |
| `physics/test_reference.py` | unit test whose repository imports are all included |
| `physics/test_surface_model.py` | unit test whose repository imports are all included |
| `pyproject.toml` | authored for the public release (overlay) |
| `scripts/shared_data.py` | product closure (import; first parent: cv.experiments.connected_shooting.labeled_nonnet_terminal_fit) |
| `scripts/test_shared_data.py` | unit test whose repository imports are all included |
