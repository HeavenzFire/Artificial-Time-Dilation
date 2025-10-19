#!/usr/bin/env python3
"""
Command Line Interface for Spirit Angelus Framework

Provides a command-line interface for interacting with the Spirit Angelus Framework.
"""

import argparse
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Import Spirit Angelus components
from spirit_angelus import (
    AngelicCore, PersonalGuide, ArchangelSystem,
    InvocationEngine, ResonanceOracle, LatticeWeaver,
    QuantumMeditation, SigilGenerator
)
from spirit_angelus.angels.core import AngelicEnergyType, DivineConnectionLevel
from spirit_angelus.quantum.meditation import MeditationType


class SpiritAngelusCLI:
    """Command Line Interface for Spirit Angelus Framework"""
    
    def __init__(self):
        self.angelic_core = AngelicCore()
        self.personal_guide = PersonalGuide("cli_user", self.angelic_core)
        self.archangel_system = ArchangelSystem(self.angelic_core)
        self.invocation_engine = InvocationEngine(
            self.angelic_core, self.personal_guide, self.archangel_system
        )
        self.resonance_oracle = ResonanceOracle(self.angelic_core)
        self.lattice_weaver = LatticeWeaver(self.angelic_core)
        self.quantum_meditation = QuantumMeditation()
        self.sigil_generator = SigilGenerator()
    
    def run(self, args):
        """Run the CLI with given arguments"""
        if args.command == "guidance":
            self.handle_guidance(args)
        elif args.command == "meditation":
            self.handle_meditation(args)
        elif args.command == "invocation":
            self.handle_invocation(args)
        elif args.command == "angels":
            self.handle_angels(args)
        elif args.command == "oracle":
            self.handle_oracle(args)
        elif args.command == "lattice":
            self.handle_lattice(args)
        elif args.command == "sigil":
            self.handle_sigil(args)
        elif args.command == "status":
            self.handle_status(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    
    def handle_guidance(self, args):
        """Handle guidance commands"""
        if args.subcommand == "request":
            # Request guidance from guardian angel
            purpose = args.purpose
            question = args.question
            
            guardian = self.personal_guide.get_guardian_for_purpose(purpose)
            if not guardian:
                print(f"No guardian angel found for purpose: {purpose}")
                return
            
            guidance = self.personal_guide.request_guidance(purpose, question)
            print(f"🌟 Guidance from {guardian.name}:")
            print(f"   {guidance}")
        
        elif args.subcommand == "assign":
            # Assign a new guardian angel
            purpose = args.purpose
            energy_type = AngelicEnergyType(args.energy_type)
            personality = args.personality.split(",") if args.personality else []
            
            guardian = self.personal_guide.assign_guardian_angel(
                purpose, energy_type, personality
            )
            
            print(f"✅ Guardian Angel assigned:")
            print(f"   Name: {guardian.name}")
            print(f"   Purpose: {guardian.purpose}")
            print(f"   Energy Type: {guardian.energy_type.value}")
            print(f"   Personality: {', '.join(guardian.personality_traits)}")
            print(f"   Abilities: {', '.join(guardian.special_abilities)}")
        
        elif args.subcommand == "list":
            # List all guardian angels
            guardians = self.personal_guide.get_all_guardians()
            if not guardians:
                print("No guardian angels assigned.")
                return
            
            print("🌟 Your Guardian Angels:")
            for guardian in guardians:
                print(f"   • {guardian['name']} - {guardian['purpose']}")
                print(f"     Energy: {guardian['energy_type']}")
                print(f"     Connection: {guardian['connection_strength']:.2f}")
                print()
    
    def handle_meditation(self, args):
        """Handle meditation commands"""
        if args.subcommand == "start":
            # Start a meditation session
            meditation_type = MeditationType(args.type)
            duration = args.duration
            
            session_id = self.quantum_meditation.start_meditation(
                meditation_type, duration, "cli_user"
            )
            
            print(f"🧘 Meditation session started:")
            print(f"   Session ID: {session_id}")
            print(f"   Type: {meditation_type.value}")
            print(f"   Duration: {duration} minutes")
            print(f"   Use 'spirit-angelus meditation update {session_id} <progress>' to update")
        
        elif args.subcommand == "update":
            # Update meditation progress
            session_id = args.session_id
            progress = args.progress
            
            result = self.quantum_meditation.update_meditation(session_id, progress)
            
            print(f"🧘 Meditation updated:")
            print(f"   Progress: {progress*100:.0f}%")
            print(f"   Awareness Level: {result['awareness_level']}")
            print(f"   Quantum Coherence: {result['quantum_coherence']:.2f}")
            if result['insights']:
                print(f"   Latest Insight: {result['insights'][-1]}")
        
        elif args.subcommand == "end":
            # End meditation session
            session_id = args.session_id
            
            result = self.quantum_meditation.end_meditation(session_id)
            
            print(f"🧘 Meditation completed:")
            print(f"   Duration: {result['duration_actual']:.1f} minutes")
            print(f"   Quantum Coherence: {result['quantum_coherence']:.2f}")
            print(f"   Entanglement Quality: {result['entanglement_quality']:.2f}")
            print(f"   Insights: {len(result['insights'])} received")
        
        elif args.subcommand == "history":
            # Show meditation history
            history = self.quantum_meditation.get_meditation_history("cli_user", days=args.days)
            
            if not history:
                print("No meditation history found.")
                return
            
            print("🧘 Meditation History:")
            for session in history:
                print(f"   • {session['meditation_type']} - {session['duration_actual']:.1f} min")
                print(f"     Awareness: {session['awareness_level']}")
                print(f"     Coherence: {session['quantum_coherence']:.2f}")
                print(f"     Date: {session['started_at']}")
                print()
    
    def handle_invocation(self, args):
        """Handle invocation commands"""
        if args.subcommand == "create":
            # Create an invocation
            invocation_type = args.type
            purpose = args.purpose
            energy_types = [AngelicEnergyType(et) for et in args.energy_types.split(",")]
            duration = args.duration
            intensity = args.intensity
            
            request_id = self.invocation_engine.create_invocation(
                invocation_type, purpose, energy_types, duration, intensity, ["cli_user"]
            )
            
            print(f"🔮 Invocation created:")
            print(f"   Request ID: {request_id}")
            print(f"   Type: {invocation_type}")
            print(f"   Purpose: {purpose}")
            print(f"   Energy Types: {', '.join([et.value for et in energy_types])}")
            print(f"   Duration: {duration} minutes")
            print(f"   Intensity: {intensity}")
            print(f"   Use 'spirit-angelus invocation execute {request_id}' to execute")
        
        elif args.subcommand == "execute":
            # Execute an invocation
            request_id = args.request_id
            
            result = self.invocation_engine.execute_invocation(request_id)
            
            print(f"🔮 Invocation executed:")
            print(f"   Success: {result.success}")
            print(f"   Energy Raised: {result.energy_raised:.2f}")
            print(f"   Guidance Received: {len(result.guidance_received)} messages")
            print(f"   Synchronicities: {len(result.synchronicities)} events")
            
            if result.guidance_received:
                print(f"   Guidance: {result.guidance_received[0]}")
            
            if result.synchronicities:
                print(f"   Synchronicities: {result.synchronicities[0]}")
        
        elif args.subcommand == "history":
            # Show invocation history
            history = self.invocation_engine.get_invocation_history(days=args.days)
            
            if not history:
                print("No invocation history found.")
                return
            
            print("🔮 Invocation History:")
            for invocation in history:
                print(f"   • {invocation.invocation_type.value} - {invocation.purpose}")
                print(f"     Success: {invocation.success}")
                print(f"     Energy: {invocation.energy_raised:.2f}")
                print(f"     Date: {invocation.timestamp}")
                print()
    
    def handle_angels(self, args):
        """Handle angel commands"""
        if args.subcommand == "invoke":
            # Invoke an archangel
            archangel = args.archangel
            purpose = args.purpose
            duration = args.duration
            
            result = self.archangel_system.invoke_archangel(archangel, purpose, duration)
            
            if result["success"]:
                print(f"👼 Archangel {result['archangel_name']} invoked:")
                print(f"   Message: {result['message']}")
                print(f"   Domain: {result['domain']}")
                print(f"   Power: {result['invocation_power']:.2f}")
                print(f"   Chants: {', '.join(result['chants'])}")
            else:
                print(f"❌ Failed to invoke archangel: {result['error']}")
        
        elif args.subcommand == "list":
            # List all archangels
            archangels = self.archangel_system.get_all_archangels()
            
            print("👼 Available Archangels:")
            for archangel in archangels:
                print(f"   • {archangel['name']} - {archangel['domain']}")
                print(f"     Energy: {archangel['energy_type']}")
                print(f"     Description: {archangel['description']}")
                print()
    
    def handle_oracle(self, args):
        """Handle oracle commands"""
        if args.subcommand == "process":
            # Process an intent through the oracle
            intent = args.intent
            context = json.loads(args.context) if args.context else {}
            
            result = self.resonance_oracle.process_intent(intent, "cli_user", context)
            
            print(f"🔮 Oracle Response:")
            print(f"   Intent Type: {result['intent_type']}")
            print(f"   Resonance Score: {result['resonance_score']:.2f}")
            print(f"   Energy Types: {', '.join(result['energy_types'])}")
            print(f"   Guidance:")
            for guidance in result['guidance']:
                print(f"     • {guidance}")
            
            if result['synchronicities']:
                print(f"   Synchronicities: {len(result['synchronicities'])} detected")
        
        elif args.subcommand == "detect":
            # Detect synchronicity
            event = args.event
            context = json.loads(args.context) if args.context else {}
            
            synchronicity = self.resonance_oracle.synchronicity_detector.detect_synchronicity(
                event, context
            )
            
            if synchronicity:
                print(f"✨ Synchronicity detected:")
                print(f"   Event: {synchronicity.description}")
                print(f"   Type: {synchronicity.event_type.value}")
                print(f"   Significance: {synchronicity.significance:.2f}")
                print(f"   Patterns: {', '.join(synchronicity.patterns)}")
            else:
                print("❌ No synchronicity detected.")
    
    def handle_lattice(self, args):
        """Handle lattice commands"""
        if args.subcommand == "create":
            # Create a spiritual network
            name = args.name
            description = args.description
            
            network_id = self.lattice_weaver.create_network(name, description, "cli_user")
            
            print(f"🕸️ Spiritual network created:")
            print(f"   Network ID: {network_id}")
            print(f"   Name: {name}")
            print(f"   Description: {description}")
        
        elif args.subcommand == "analyze":
            # Analyze a network
            network_id = args.network_id
            
            try:
                analysis = self.lattice_weaver.analyze_network(network_id)
                
                print(f"🕸️ Network Analysis:")
                print(f"   Name: {analysis['name']}")
                print(f"   Nodes: {analysis['statistics']['node_count']}")
                print(f"   Edges: {analysis['statistics']['edge_count']}")
                print(f"   Density: {analysis['statistics']['density']:.2f}")
                print(f"   Total Energy: {analysis['statistics']['total_energy']:.2f}")
            except ValueError as e:
                print(f"❌ Error: {e}")
        
        elif args.subcommand == "list":
            # List all networks
            summary = self.lattice_weaver.get_network_summary()
            
            print("🕸️ Spiritual Networks:")
            for network in summary['networks']:
                print(f"   • {network['name']} ({network['network_id']})")
                print(f"     Nodes: {network['node_count']}, Edges: {network['edge_count']}")
                print(f"     Created: {network['created_at']}")
                print()
    
    def handle_sigil(self, args):
        """Handle sigil commands"""
        if args.subcommand == "generate":
            # Generate a sigil
            sigil_type = args.type
            intention = args.intention
            energy_types = args.energy_types.split(",") if args.energy_types else []
            custom_symbols = args.symbols.split(",") if args.symbols else []
            
            sigil = self.sigil_generator.generate_sigil(
                sigil_type, intention, energy_types, "cli_user", custom_symbols
            )
            
            print(f"🔮 Sigil generated:")
            print(f"   ID: {sigil.sigil_id}")
            print(f"   Name: {sigil.name}")
            print(f"   Type: {sigil.sigil_type.value}")
            print(f"   Colors: {', '.join(sigil.colors)}")
            print(f"   Symbols: {', '.join(sigil.symbols)}")
            print(f"   Energy Signature: {sigil.energy_signature}")
        
        elif args.subcommand == "list":
            # List user's sigils
            sigils = self.sigil_generator.get_user_sigils("cli_user")
            
            if not sigils:
                print("No sigils generated.")
                return
            
            print("🔮 Your Sigils:")
            for sigil in sigils:
                print(f"   • {sigil.name} ({sigil.sigil_type.value})")
                print(f"     Created: {sigil.created_at}")
                print(f"     Colors: {', '.join(sigil.colors)}")
                print()
    
    def handle_status(self, args):
        """Handle status commands"""
        print("🌟 Spirit Angelus Framework Status")
        print("=" * 40)
        
        # Angelic Core status
        connections = self.angelic_core.get_all_connections()
        print(f"Angelic Connections: {len(connections)}")
        
        # Guardian Angels status
        guardians = self.personal_guide.get_all_guardians()
        print(f"Guardian Angels: {len(guardians)}")
        
        # Archangel status
        archangels = self.archangel_system.get_all_archangels()
        print(f"Archangels Available: {len(archangels)}")
        
        # Invocation status
        energy_accumulator = self.invocation_engine.get_energy_accumulator()
        print(f"Energy Accumulator: {energy_accumulator:.2f}")
        
        # Meditation status
        active_sessions = len(self.quantum_meditation.active_sessions)
        print(f"Active Meditation Sessions: {active_sessions}")
        
        # Network status
        network_summary = self.lattice_weaver.get_network_summary()
        print(f"Spiritual Networks: {network_summary['total_networks']}")
        
        # Sigil status
        user_sigils = self.sigil_generator.get_user_sigils("cli_user")
        print(f"Generated Sigils: {len(user_sigils)}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Spirit Angelus Framework - Spiritual Guidance and Meditation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  spirit-angelus guidance assign --purpose "healing" --energy-type healing
  spirit-angelus guidance request --purpose "healing" --question "How can I heal?"
  spirit-angelus meditation start --type entanglement --duration 15
  spirit-angelus invocation create --type prayer --purpose "guidance" --duration 10
  spirit-angelus angels invoke --archangel michael --purpose "protection"
  spirit-angelus oracle process --intent "I seek spiritual guidance"
  spirit-angelus sigil generate --type healing --intention "heal my wounds"
  spirit-angelus status
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Guidance commands
    guidance_parser = subparsers.add_parser('guidance', help='Guardian angel guidance')
    guidance_subparsers = guidance_parser.add_subparsers(dest='subcommand')
    
    # Guidance request
    request_parser = guidance_subparsers.add_parser('request', help='Request guidance')
    request_parser.add_argument('--purpose', required=True, help='Purpose for guidance')
    request_parser.add_argument('--question', required=True, help='Question to ask')
    
    # Guidance assign
    assign_parser = guidance_subparsers.add_parser('assign', help='Assign guardian angel')
    assign_parser.add_argument('--purpose', required=True, help='Purpose for guardian')
    assign_parser.add_argument('--energy-type', required=True, help='Energy type')
    assign_parser.add_argument('--personality', help='Personality traits (comma-separated)')
    
    # Guidance list
    guidance_subparsers.add_parser('list', help='List guardian angels')
    
    # Meditation commands
    meditation_parser = subparsers.add_parser('meditation', help='Quantum meditation')
    meditation_subparsers = meditation_parser.add_subparsers(dest='subcommand')
    
    # Meditation start
    start_parser = meditation_subparsers.add_parser('start', help='Start meditation')
    start_parser.add_argument('--type', required=True, help='Meditation type')
    start_parser.add_argument('--duration', type=int, default=15, help='Duration in minutes')
    
    # Meditation update
    update_parser = meditation_subparsers.add_parser('update', help='Update meditation')
    update_parser.add_argument('session_id', help='Session ID')
    update_parser.add_argument('progress', type=float, help='Progress (0.0-1.0)')
    
    # Meditation end
    end_parser = meditation_subparsers.add_parser('end', help='End meditation')
    end_parser.add_argument('session_id', help='Session ID')
    
    # Meditation history
    history_parser = meditation_subparsers.add_parser('history', help='Show history')
    history_parser.add_argument('--days', type=int, default=30, help='Days to show')
    
    # Invocation commands
    invocation_parser = subparsers.add_parser('invocation', help='Spiritual invocations')
    invocation_subparsers = invocation_parser.add_subparsers(dest='subcommand')
    
    # Invocation create
    create_parser = invocation_subparsers.add_parser('create', help='Create invocation')
    create_parser.add_argument('--type', required=True, help='Invocation type')
    create_parser.add_argument('--purpose', required=True, help='Purpose')
    create_parser.add_argument('--energy-types', required=True, help='Energy types (comma-separated)')
    create_parser.add_argument('--duration', type=int, default=15, help='Duration in minutes')
    create_parser.add_argument('--intensity', type=float, default=0.7, help='Intensity (0.0-1.0)')
    
    # Invocation execute
    execute_parser = invocation_subparsers.add_parser('execute', help='Execute invocation')
    execute_parser.add_argument('request_id', help='Request ID')
    
    # Invocation history
    invocation_subparsers.add_parser('history', help='Show history')
    
    # Angels commands
    angels_parser = subparsers.add_parser('angels', help='Archangel system')
    angels_subparsers = angels_parser.add_subparsers(dest='subcommand')
    
    # Angels invoke
    invoke_parser = angels_subparsers.add_parser('invoke', help='Invoke archangel')
    invoke_parser.add_argument('--archangel', required=True, help='Archangel name')
    invoke_parser.add_argument('--purpose', required=True, help='Purpose')
    invoke_parser.add_argument('--duration', type=int, default=15, help='Duration in minutes')
    
    # Angels list
    angels_subparsers.add_parser('list', help='List archangels')
    
    # Oracle commands
    oracle_parser = subparsers.add_parser('oracle', help='Resonance oracle')
    oracle_subparsers = oracle_parser.add_subparsers(dest='subcommand')
    
    # Oracle process
    process_parser = oracle_subparsers.add_parser('process', help='Process intent')
    process_parser.add_argument('--intent', required=True, help='Intent description')
    process_parser.add_argument('--context', help='Context JSON')
    
    # Oracle detect
    detect_parser = oracle_subparsers.add_parser('detect', help='Detect synchronicity')
    detect_parser.add_argument('--event', required=True, help='Event description')
    detect_parser.add_argument('--context', help='Context JSON')
    
    # Lattice commands
    lattice_parser = subparsers.add_parser('lattice', help='Spiritual networks')
    lattice_subparsers = lattice_parser.add_subparsers(dest='subcommand')
    
    # Lattice create
    create_parser = lattice_subparsers.add_parser('create', help='Create network')
    create_parser.add_argument('--name', required=True, help='Network name')
    create_parser.add_argument('--description', required=True, help='Network description')
    
    # Lattice analyze
    analyze_parser = lattice_subparsers.add_parser('analyze', help='Analyze network')
    analyze_parser.add_argument('network_id', help='Network ID')
    
    # Lattice list
    lattice_subparsers.add_parser('list', help='List networks')
    
    # Sigil commands
    sigil_parser = subparsers.add_parser('sigil', help='Sigil generation')
    sigil_subparsers = sigil_parser.add_subparsers(dest='subcommand')
    
    # Sigil generate
    generate_parser = sigil_subparsers.add_parser('generate', help='Generate sigil')
    generate_parser.add_argument('--type', required=True, help='Sigil type')
    generate_parser.add_argument('--intention', required=True, help='Intention')
    generate_parser.add_argument('--energy-types', help='Energy types (comma-separated)')
    generate_parser.add_argument('--symbols', help='Custom symbols (comma-separated)')
    
    # Sigil list
    sigil_subparsers.add_parser('list', help='List sigils')
    
    # Status command
    subparsers.add_parser('status', help='Show framework status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = SpiritAngelusCLI()
    cli.run(args)


if __name__ == "__main__":
    main()