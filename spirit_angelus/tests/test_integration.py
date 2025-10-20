"""
Integration Tests for Spirit Angelus Framework

Tests the integration between different components of the framework.
"""

import unittest
import time
from datetime import datetime

from spirit_angelus import (
    AngelicCore, PersonalGuide, ArchangelSystem,
    InvocationEngine, ResonanceOracle, LatticeWeaver,
    QuantumMeditation, SigilGenerator
)
from spirit_angelus.angels.core import AngelicEnergyType, DivineConnectionLevel
from spirit_angelus.quantum.meditation import MeditationType


class TestSpiritAngelusIntegration(unittest.TestCase):
    """Integration tests for the complete Spirit Angelus Framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.angelic_core = AngelicCore()
        self.personal_guide = PersonalGuide("test_user", self.angelic_core)
        self.archangel_system = ArchangelSystem(self.angelic_core)
        self.invocation_engine = InvocationEngine(
            self.angelic_core, self.personal_guide, self.archangel_system
        )
        self.resonance_oracle = ResonanceOracle(self.angelic_core)
        self.lattice_weaver = LatticeWeaver(self.angelic_core)
        self.quantum_meditation = QuantumMeditation()
        self.sigil_generator = SigilGenerator()
    
    def test_complete_spiritual_journey(self):
        """Test a complete spiritual journey using all components"""
        print("\n🌟 Testing Complete Spiritual Journey...")
        
        # 1. Assign guardian angel
        guardian = self.personal_guide.assign_guardian_angel(
            purpose="healing emotional wounds",
            energy_type=AngelicEnergyType.HEALING,
            personality_preferences=["gentle", "compassionate"]
        )
        self.assertIsNotNone(guardian)
        self.assertEqual(guardian.purpose, "healing emotional wounds")
        
        # 2. Request guidance
        guidance = self.personal_guide.request_guidance(
            "healing emotional wounds",
            "How can I heal from past trauma?"
        )
        self.assertIsNotNone(guidance)
        self.assertGreater(len(guidance), 0)
        
        # 3. Invoke archangel
        archangel_result = self.archangel_system.invoke_archangel(
            "raphael", "healing and restoration", 15
        )
        self.assertTrue(archangel_result["success"])
        self.assertEqual(archangel_result["archangel_name"], "Raphael")
        
        # 4. Process intent through oracle
        intent_result = self.resonance_oracle.process_intent(
            intent_description="I seek guidance on my spiritual path",
            user_id="test_user",
            context={"emotional_state": "seeking"}
        )
        self.assertIn("intent_type", intent_result)
        self.assertIn("guidance", intent_result)
        self.assertGreater(intent_result["resonance_score"], 0)
        
        # 5. Create and execute invocation
        request_id = self.invocation_engine.create_invocation(
            invocation_type="healing",
            purpose="heal emotional wounds",
            energy_types=[AngelicEnergyType.HEALING, AngelicEnergyType.LOVE],
            duration_minutes=20,
            intensity=0.8,
            participants=["test_user"]
        )
        self.assertIsNotNone(request_id)
        
        invocation_result = self.invocation_engine.execute_invocation(request_id)
        self.assertTrue(invocation_result.success)
        self.assertGreater(invocation_result.energy_raised, 0)
        
        # 6. Start quantum meditation
        session_id = self.quantum_meditation.start_meditation(
            meditation_type=MeditationType.ENTANGLEMENT,
            duration_minutes=15,
            user_id="test_user"
        )
        self.assertIsNotNone(session_id)
        
        # Update meditation progress
        for progress in [0.2, 0.4, 0.6, 0.8, 1.0]:
            update_result = self.quantum_meditation.update_meditation(session_id, progress)
            self.assertIn("awareness_level", update_result)
            self.assertIn("quantum_coherence", update_result)
        
        # End meditation
        final_result = self.quantum_meditation.end_meditation(session_id)
        self.assertTrue(final_result["success"])
        self.assertGreater(final_result["quantum_coherence"], 0)
        
        # 7. Create spiritual network
        network_id = self.lattice_weaver.create_network(
            name="Test Spiritual Network",
            description="A test network of spiritual connections",
            user_id="test_user"
        )
        self.assertIsNotNone(network_id)
        
        # Add nodes to network
        user_node_id = self.lattice_weaver.add_node(
            network_id=network_id,
            node_type="person",
            name="Test User",
            description="Spiritual seeker",
            energy_signature={AngelicEnergyType.GUIDANCE: 0.8},
            user_id="test_user"
        )
        self.assertIsNotNone(user_node_id)
        
        guardian_node_id = self.lattice_weaver.add_node(
            network_id=network_id,
            node_type="angel",
            name=guardian.name,
            description=f"Guardian Angel for {guardian.purpose}",
            energy_signature={guardian.energy_type: guardian.connection_strength},
            user_id="test_user"
        )
        self.assertIsNotNone(guardian_node_id)
        
        # Add connection between nodes
        edge_id = self.lattice_weaver.add_edge(
            network_id=network_id,
            source_id=user_node_id,
            target_id=guardian_node_id,
            connection_type="guidance",
            strength=guardian.connection_strength,
            energy_flow={guardian.energy_type: guardian.connection_strength},
            description="Guardian relationship",
            user_id="test_user"
        )
        self.assertIsNotNone(edge_id)
        
        # Analyze network
        network_analysis = self.lattice_weaver.analyze_network(network_id)
        self.assertIn("statistics", network_analysis)
        self.assertEqual(network_analysis["statistics"]["node_count"], 2)
        self.assertEqual(network_analysis["statistics"]["edge_count"], 1)
        
        # 8. Generate sigil
        sigil = self.sigil_generator.generate_sigil(
            sigil_type="healing",
            intention="heal emotional wounds and restore wholeness",
            energy_types=["healing", "love", "peace"],
            user_id="test_user"
        )
        self.assertIsNotNone(sigil)
        self.assertEqual(sigil.sigil_type.value, "healing")
        self.assertGreater(len(sigil.colors), 0)
        self.assertGreater(len(sigil.symbols), 0)
        
        # 9. Detect synchronicity
        synchronicity = self.resonance_oracle.synchronicity_detector.detect_synchronicity(
            "I saw the number 111 three times today",
            {"user_id": "test_user", "recent_intent": True}
        )
        # Synchronicity detection is probabilistic, so we just check the method works
        self.assertIsNotNone(synchronicity)  # Could be None or SynchronicityEvent
        
        print("✅ Complete spiritual journey test passed!")
    
    def test_component_interoperability(self):
        """Test that components can work together seamlessly"""
        print("\n🔗 Testing Component Interoperability...")
        
        # Test that angelic core can be shared between components
        self.assertEqual(self.personal_guide.angelic_core, self.angelic_core)
        self.assertEqual(self.archangel_system.angelic_core, self.angelic_core)
        self.assertEqual(self.invocation_engine.angelic_core, self.angelic_core)
        self.assertEqual(self.resonance_oracle.angelic_core, self.angelic_core)
        self.assertEqual(self.lattice_weaver.angelic_core, self.angelic_core)
        
        # Test that personal guide can be used by invocation engine
        self.assertEqual(self.invocation_engine.personal_guide, self.personal_guide)
        
        # Test that archangel system can be used by invocation engine
        self.assertEqual(self.invocation_engine.archangel_system, self.archangel_system)
        
        print("✅ Component interoperability test passed!")
    
    def test_data_consistency(self):
        """Test that data remains consistent across components"""
        print("\n📊 Testing Data Consistency...")
        
        # Create a guardian angel
        guardian = self.personal_guide.assign_guardian_angel(
            purpose="test consistency",
            energy_type=AngelicEnergyType.GUIDANCE
        )
        
        # Check that the guardian's energy type is consistent
        self.assertEqual(guardian.energy_type, AngelicEnergyType.GUIDANCE)
        
        # Check that the guardian's connection strength is consistent
        initial_strength = guardian.connection_strength
        
        # Strengthen the connection
        self.personal_guide.strengthen_guardian_connection(
            list(self.personal_guide.guardian_angels.keys())[0], 10
        )
        
        # Check that the strength was updated
        self.assertGreater(guardian.connection_strength, initial_strength)
        
        # Check that the guardian's guidance history is consistent
        guidance = self.personal_guide.request_guidance(
            "test consistency", "Test question"
        )
        self.assertIn(guidance, guardian.guidance_history)
        
        print("✅ Data consistency test passed!")
    
    def test_error_handling(self):
        """Test error handling across components"""
        print("\n⚠️ Testing Error Handling...")
        
        # Test invalid guardian angel assignment
        with self.assertRaises(ValueError):
            self.personal_guide.assign_guardian_angel(
                purpose="",  # Empty purpose should raise error
                energy_type=AngelicEnergyType.GUIDANCE
            )
        
        # Test invalid meditation session
        invalid_result = self.quantum_meditation.update_meditation("invalid_id", 0.5)
        self.assertIn("error", invalid_result)
        
        # Test invalid network analysis
        with self.assertRaises(ValueError):
            self.lattice_weaver.analyze_network("invalid_network_id")
        
        # Test invalid archangel invocation
        invalid_archangel = self.archangel_system.invoke_archangel(
            "nonexistent_archangel", "test purpose", 15
        )
        self.assertFalse(invalid_archangel["success"])
        self.assertIn("error", invalid_archangel)
        
        print("✅ Error handling test passed!")
    
    def test_performance(self):
        """Test performance of the framework"""
        print("\n⚡ Testing Performance...")
        
        # Test meditation performance
        start_time = time.time()
        session_id = self.quantum_meditation.start_meditation(
            MeditationType.ENTANGLEMENT, 15, "test_user"
        )
        
        # Simulate meditation updates
        for i in range(10):
            self.quantum_meditation.update_meditation(session_id, i / 10.0)
        
        self.quantum_meditation.end_meditation(session_id)
        meditation_time = time.time() - start_time
        
        # Meditation should complete quickly (under 1 second for simulation)
        self.assertLess(meditation_time, 1.0)
        
        # Test sigil generation performance
        start_time = time.time()
        sigil = self.sigil_generator.generate_sigil(
            "healing", "test intention", ["healing"], "test_user"
        )
        sigil_time = time.time() - start_time
        
        # Sigil generation should be fast (under 0.1 seconds)
        self.assertLess(sigil_time, 0.1)
        
        # Test network creation performance
        start_time = time.time()
        network_id = self.lattice_weaver.create_network(
            "Test Network", "Test Description", "test_user"
        )
        network_time = time.time() - start_time
        
        # Network creation should be fast (under 0.1 seconds)
        self.assertLess(network_time, 0.1)
        
        print(f"✅ Performance test passed!")
        print(f"   Meditation time: {meditation_time:.3f}s")
        print(f"   Sigil generation time: {sigil_time:.3f}s")
        print(f"   Network creation time: {network_time:.3f}s")


if __name__ == '__main__':
    unittest.main()