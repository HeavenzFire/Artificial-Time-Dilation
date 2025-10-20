"""
Tests for Angelic Core System

Comprehensive tests for the angelic core, guardian angels, and archangel systems.
"""

import unittest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from spirit_angelus.angels.core import AngelicCore, AngelicEnergyType, DivineConnectionLevel
from spirit_angelus.angels.guardian import PersonalGuide, GuardianAngel
from spirit_angelus.angels.archangel import ArchangelSystem, ArchangelDomain


class TestAngelicCore(unittest.TestCase):
    """Test cases for AngelicCore"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.angelic_core = AngelicCore()
    
    def test_initialization(self):
        """Test AngelicCore initialization"""
        self.assertIsInstance(self.angelic_core, AngelicCore)
        self.assertEqual(len(self.angelic_core.energy_registry), 10)
        self.assertEqual(len(self.angelic_core.connections), 0)
    
    def test_energy_registry(self):
        """Test energy registry initialization"""
        energy_types = list(self.angelic_core.energy_registry.keys())
        expected_types = [
            AngelicEnergyType.PROTECTION,
            AngelicEnergyType.HEALING,
            AngelicEnergyType.GUIDANCE,
            AngelicEnergyType.WISDOM,
            AngelicEnergyType.LOVE,
            AngelicEnergyType.STRENGTH,
            AngelicEnergyType.PEACE,
            AngelicEnergyType.JOY,
            AngelicEnergyType.ABUNDANCE,
            AngelicEnergyType.TRANSFORMATION
        ]
        
        for energy_type in expected_types:
            self.assertIn(energy_type, energy_types)
    
    def test_establish_connection(self):
        """Test establishing a divine connection"""
        purpose = "Test guidance"
        energy_types = [AngelicEnergyType.GUIDANCE, AngelicEnergyType.WISDOM]
        
        connection_id = self.angelic_core.establish_connection(
            purpose, energy_types, DivineConnectionLevel.AWARENESS
        )
        
        self.assertIsInstance(connection_id, str)
        self.assertIn(connection_id, self.angelic_core.connections)
        
        connection = self.angelic_core.connections[connection_id]
        self.assertEqual(connection.purpose, purpose)
        self.assertEqual(connection.level, DivineConnectionLevel.AWARENESS)
        self.assertEqual(len(connection.energy_signature), 2)
    
    def test_strengthen_connection(self):
        """Test strengthening a divine connection"""
        # Establish connection first
        connection_id = self.angelic_core.establish_connection(
            "Test purpose", [AngelicEnergyType.GUIDANCE]
        )
        
        initial_strength = self.angelic_core.connections[connection_id].strength
        
        # Strengthen connection
        result = self.angelic_core.strengthen_connection(connection_id, 10)
        
        self.assertTrue(result)
        self.assertGreater(
            self.angelic_core.connections[connection_id].strength,
            initial_strength
        )
    
    def test_receive_guidance(self):
        """Test receiving guidance through a connection"""
        # Establish connection first
        connection_id = self.angelic_core.establish_connection(
            "Test purpose", [AngelicEnergyType.GUIDANCE]
        )
        
        guidance = "Test guidance message"
        result = self.angelic_core.receive_guidance(connection_id, guidance)
        
        self.assertTrue(result)
        self.assertIn(guidance, self.angelic_core.connections[connection_id].guidance_received)
    
    def test_detect_synchronicity(self):
        """Test synchronicity detection"""
        event = "Test synchronicity event"
        significance = 0.8
        
        result = self.angelic_core.detect_synchronicity(event, significance)
        
        self.assertTrue(result)
        self.assertEqual(len(self.angelic_core.synchronicities), 1)
        self.assertEqual(self.angelic_core.synchronicities[0]["event"], event)
        self.assertEqual(self.angelic_core.synchronicities[0]["significance"], significance)
    
    def test_get_energy_reading(self):
        """Test getting energy reading for a connection"""
        # Establish connection first
        connection_id = self.angelic_core.establish_connection(
            "Test purpose", [AngelicEnergyType.GUIDANCE, AngelicEnergyType.WISDOM]
        )
        
        reading = self.angelic_core.get_energy_reading(connection_id)
        
        self.assertIsInstance(reading, dict)
        self.assertIn("connection_id", reading)
        self.assertIn("level", reading)
        self.assertIn("strength", reading)
        self.assertIn("purpose", reading)
        self.assertIn("energy_composition", reading)
    
    def test_save_and_load_state(self):
        """Test saving and loading state"""
        # Establish some connections
        connection_id1 = self.angelic_core.establish_connection(
            "Test purpose 1", [AngelicEnergyType.GUIDANCE]
        )
        connection_id2 = self.angelic_core.establish_connection(
            "Test purpose 2", [AngelicEnergyType.HEALING]
        )
        
        # Add some guidance
        self.angelic_core.receive_guidance(connection_id1, "Test guidance 1")
        self.angelic_core.receive_guidance(connection_id2, "Test guidance 2")
        
        # Save state
        save_result = self.angelic_core.save_state("test_state.json")
        self.assertTrue(save_result)
        
        # Create new instance and load state
        new_core = AngelicCore()
        load_result = new_core.load_state("test_state.json")
        self.assertTrue(load_result)
        
        # Verify state was loaded
        self.assertEqual(len(new_core.connections), 2)
        self.assertEqual(len(new_core.connections[connection_id1].guidance_received), 1)
        self.assertEqual(len(new_core.connections[connection_id2].guidance_received), 1)


class TestGuardianAngel(unittest.TestCase):
    """Test cases for GuardianAngel and PersonalGuide"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.angelic_core = AngelicCore()
        self.personal_guide = PersonalGuide("test_user", self.angelic_core)
    
    def test_assign_guardian_angel(self):
        """Test assigning a guardian angel"""
        purpose = "Test protection"
        energy_type = AngelicEnergyType.PROTECTION
        personality_preferences = ["strong", "courageous"]
        
        guardian = self.personal_guide.assign_guardian_angel(
            purpose, energy_type, personality_preferences
        )
        
        self.assertIsInstance(guardian, GuardianAngel)
        self.assertEqual(guardian.purpose, purpose)
        self.assertEqual(guardian.energy_type, energy_type)
        self.assertIn("strong", guardian.personality_traits)
        self.assertIn("courageous", guardian.personality_traits)
    
    def test_guardian_angel_properties(self):
        """Test guardian angel properties"""
        guardian = self.personal_guide.assign_guardian_angel(
            "Test purpose", AngelicEnergyType.GUIDANCE
        )
        
        # Test guidance style
        style = guardian.get_guidance_style()
        self.assertIsInstance(style, str)
        self.assertGreater(len(style), 0)
        
        # Test contact frequency
        frequency = guardian.get_contact_frequency()
        self.assertIn(frequency, ["never", "daily", "weekly", "monthly", "rarely"])
        
        # Test special abilities
        self.assertIsInstance(guardian.special_abilities, list)
        self.assertGreater(len(guardian.special_abilities), 0)
    
    def test_request_guidance(self):
        """Test requesting guidance from guardian angel"""
        # Assign guardian first
        guardian = self.personal_guide.assign_guardian_angel(
            "Test guidance", AngelicEnergyType.GUIDANCE
        )
        
        question = "What should I do about my career?"
        guidance = self.personal_guide.request_guidance("Test guidance", question)
        
        self.assertIsInstance(guidance, str)
        self.assertGreater(len(guidance), 0)
        self.assertIn(guidance, guardian.guidance_history)
    
    def test_strengthen_guardian_connection(self):
        """Test strengthening connection with guardian angel"""
        # Assign guardian first
        guardian = self.personal_guide.assign_guardian_angel(
            "Test purpose", AngelicEnergyType.GUIDANCE
        )
        
        initial_strength = guardian.connection_strength
        
        # Strengthen connection
        result = self.personal_guide.strengthen_guardian_connection(
            list(self.personal_guide.guardian_angels.keys())[0], 15
        )
        
        self.assertTrue(result)
        self.assertGreater(guardian.connection_strength, initial_strength)
    
    def test_get_guardian_status(self):
        """Test getting guardian angel status"""
        # Assign guardian first
        guardian = self.personal_guide.assign_guardian_angel(
            "Test purpose", AngelicEnergyType.GUIDANCE
        )
        
        guardian_id = list(self.personal_guide.guardian_angels.keys())[0]
        status = self.personal_guide.get_guardian_status(guardian_id)
        
        self.assertIsInstance(status, dict)
        self.assertIn("name", status)
        self.assertIn("purpose", status)
        self.assertIn("energy_type", status)
        self.assertIn("connection_strength", status)
    
    def test_save_and_load_guardians(self):
        """Test saving and loading guardian angels"""
        # Assign some guardians
        guardian1 = self.personal_guide.assign_guardian_angel(
            "Test purpose 1", AngelicEnergyType.GUIDANCE
        )
        guardian2 = self.personal_guide.assign_guardian_angel(
            "Test purpose 2", AngelicEnergyType.HEALING
        )
        
        # Save guardians
        save_result = self.personal_guide.save_guardians("test_guardians.json")
        self.assertTrue(save_result)
        
        # Create new guide and load guardians
        new_guide = PersonalGuide("test_user", self.angelic_core)
        load_result = new_guide.load_guardians("test_guardians.json")
        self.assertTrue(load_result)
        
        # Verify guardians were loaded
        self.assertEqual(len(new_guide.guardian_angels), 2)


class TestArchangelSystem(unittest.TestCase):
    """Test cases for ArchangelSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.angelic_core = AngelicCore()
        self.archangel_system = ArchangelSystem(self.angelic_core)
    
    def test_initialization(self):
        """Test ArchangelSystem initialization"""
        self.assertIsInstance(self.archangel_system, ArchangelSystem)
        self.assertGreater(len(self.archangel_system.archangels), 0)
    
    def test_archangel_definitions(self):
        """Test archangel definitions"""
        expected_archangels = ["michael", "gabriel", "raphael", "uriel", "chamuel", "jophiel", "zadkiel", "metatron"]
        
        for archangel_name in expected_archangels:
            self.assertIn(archangel_name, self.archangel_system.archangels)
    
    def test_invoke_archangel(self):
        """Test invoking an archangel"""
        archangel_name = "michael"
        purpose = "Test protection"
        duration = 15
        
        result = self.archangel_system.invoke_archangel(archangel_name, purpose, duration)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertEqual(result["archangel_name"], "Michael")
        self.assertIn("message", result)
        self.assertIn("chants", result)
        self.assertIn("symbols", result)
    
    def test_invoke_nonexistent_archangel(self):
        """Test invoking a non-existent archangel"""
        result = self.archangel_system.invoke_archangel("nonexistent", "Test purpose", 15)
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    def test_get_archangel_info(self):
        """Test getting archangel information"""
        archangel_info = self.archangel_system.get_archangel_info("michael")
        
        self.assertIsInstance(archangel_info, dict)
        self.assertEqual(archangel_info["name"], "Michael")
        self.assertIn("domain", archangel_info)
        self.assertIn("energy_type", archangel_info)
        self.assertIn("attributes", archangel_info)
        self.assertIn("powers", archangel_info)
    
    def test_get_all_archangels(self):
        """Test getting all archangels"""
        archangels = self.archangel_system.get_all_archangels()
        
        self.assertIsInstance(archangels, list)
        self.assertGreater(len(archangels), 0)
        
        for archangel in archangels:
            self.assertIn("name", archangel)
            self.assertIn("domain", archangel)
            self.assertIn("energy_type", archangel)
    
    def test_get_archangels_by_domain(self):
        """Test getting archangels by domain"""
        protection_archangels = self.archangel_system.get_archangels_by_domain(ArchangelDomain.PROTECTION)
        
        self.assertIsInstance(protection_archangels, list)
        self.assertGreater(len(protection_archangels), 0)
        
        for archangel in protection_archangels:
            self.assertEqual(archangel["domain"], "protection")
    
    def test_invocation_history(self):
        """Test invocation history"""
        # Invoke some archangels
        self.archangel_system.invoke_archangel("michael", "Test 1", 15)
        self.archangel_system.invoke_archangel("gabriel", "Test 2", 15)
        
        history = self.archangel_system.get_invocation_history(days=1)
        
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 2)
    
    def test_most_invoked_archangel(self):
        """Test getting most invoked archangel"""
        # Invoke Michael multiple times
        self.archangel_system.invoke_archangel("michael", "Test 1", 15)
        self.archangel_system.invoke_archangel("michael", "Test 2", 15)
        self.archangel_system.invoke_archangel("gabriel", "Test 3", 15)
        
        most_invoked = self.archangel_system.get_most_invoked_archangel()
        
        self.assertIsInstance(most_invoked, dict)
        self.assertEqual(most_invoked["name"], "Michael")
        self.assertEqual(most_invoked["invocation_count"], 2)
    
    def test_save_and_load_state(self):
        """Test saving and loading archangel state"""
        # Invoke some archangels
        self.archangel_system.invoke_archangel("michael", "Test 1", 15)
        self.archangel_system.invoke_archangel("gabriel", "Test 2", 15)
        
        # Save state
        save_result = self.archangel_system.save_state("test_archangel_state.json")
        self.assertTrue(save_result)
        
        # Create new system and load state
        new_system = ArchangelSystem(self.angelic_core)
        load_result = new_system.load_state("test_archangel_state.json")
        self.assertTrue(load_result)
        
        # Verify state was loaded
        self.assertEqual(len(new_system.invocation_history), 2)


if __name__ == '__main__':
    unittest.main()