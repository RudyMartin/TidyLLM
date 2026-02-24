29/29 PASSED

  tests/test_humanoid_vla.py::TestPhaseDetection::test_free_motion_detected PASSED
  tests/test_humanoid_vla.py::TestPhaseDetection::test_approaching_detected PASSED
  tests/test_humanoid_vla.py::TestPhaseDetection::test_contact_detected PASSED
  tests/test_humanoid_vla.py::TestPhaseDetection::test_retreat_requested PASSED
  tests/test_humanoid_vla.py::TestPhaseDetection::test_phase_hysteresis PASSED
  tests/test_humanoid_vla.py::TestPhaseDetection::test_phase_info_provides_recommendations PASSED
  tests/test_humanoid_vla.py::TestImpedanceSelection::test_contact_phase_has_low_stiffness PASSED
  tests/test_humanoid_vla.py::TestImpedanceSelection::test_higher_quality_allows_higher_stiffness PASSED
  tests/test_humanoid_vla.py::TestImpedanceSelection::test_impedance_within_limits PASSED
  tests/test_humanoid_vla.py::TestGaitScaling::test_high_quality_full_gait PASSED
  tests/test_humanoid_vla.py::TestGaitScaling::test_low_quality_reduced_gait PASSED
  tests/test_humanoid_vla.py::TestGaitScaling::test_very_low_quality_stops PASSED
  tests/test_humanoid_vla.py::TestGaitScaling::test_gait_mode_names PASSED
  tests/test_humanoid_vla.py::TestSubsystemRouter::test_h1_joint_mapping PASSED
  tests/test_humanoid_vla.py::TestSubsystemRouter::test_command_routing PASSED
  tests/test_humanoid_vla.py::TestMockHumanoidAdapter::test_basic_state PASSED
  tests/test_humanoid_vla.py::TestMockHumanoidAdapter::test_command_execution PASSED
  tests/test_humanoid_vla.py::TestMockHumanoidAdapter::test_emergency_stop PASSED
  tests/test_humanoid_vla.py::TestMockHumanoidAdapter::test_contact_simulation PASSED
  tests/test_humanoid_vla.py::TestVLAQualityNode::test_lifecycle PASSED
  tests/test_humanoid_vla.py::TestVLAQualityNode::test_tick_processing PASSED
  tests/test_humanoid_vla.py::TestVLAQualityNode::test_phase_detection_during_tick PASSED
  tests/test_humanoid_vla.py::TestVLAQualityNode::test_low_quality_triggers_caution PASSED
  tests/test_humanoid_vla.py::TestVLAQualityNode::test_safety_violation_stops_robot PASSED
  tests/test_humanoid_vla.py::TestYRSNIntegration::test_decompose_robotics_instruction PASSED
  tests/test_humanoid_vla.py::TestYRSNIntegration::test_control_barrier_integration PASSED
  tests/test_humanoid_vla.py::TestYRSNIntegration::test_feasibility_filter_integration PASSED
  tests/test_humanoid_vla.py::TestPhaseAlphaThresholds::test_contact_phase_has_lower_threshold PASSED
  tests/test_humanoid_vla.py::TestPhaseAlphaThresholds::test_phase_impedance_values PASSED

    Coverage for new humanoid modules:
  - mock_humanoid.py: 85%
  - phase_detector.py: 85%
  - vla_quality_node.py: 77%
  - impedance_selector.py: 73%
  - subsystem_router.py: 70%
