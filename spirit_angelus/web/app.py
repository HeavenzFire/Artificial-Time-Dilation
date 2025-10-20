"""
Spirit Angelus Web Application

Main web application for the Spirit Angelus Framework, providing a modern
web interface for spiritual guidance and meditation.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import os
from datetime import datetime, timedelta
import uuid

# Import Spirit Angelus components
from ..angels.core import AngelicCore
from ..angels.guardian import PersonalGuide
from ..angels.archangel import ArchangelSystem
from ..invocation.engine import InvocationEngine
from ..oracle.oracle import ResonanceOracle
from ..lattice.weaver import LatticeWeaver
from ..quantum.meditation import QuantumMeditation


class SpiritAngelusApp:
    """
    Main web application class for the Spirit Angelus Framework.
    
    Provides a comprehensive web interface for spiritual guidance,
    meditation, and angelic connection.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.app = Flask(__name__)
        self.app.secret_key = self.config.get('secret_key', 'spirit-angelus-secret-key')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize Spirit Angelus components
        self.angelic_core = AngelicCore()
        self.personal_guide = PersonalGuide("web_user", self.angelic_core)
        self.archangel_system = ArchangelSystem(self.angelic_core)
        self.invocation_engine = InvocationEngine(
            self.angelic_core, 
            self.personal_guide, 
            self.archangel_system
        )
        self.resonance_oracle = ResonanceOracle(self.angelic_core)
        self.lattice_weaver = LatticeWeaver(self.angelic_core)
        self.quantum_meditation = QuantumMeditation()
        
        # Setup routes
        self._setup_routes()
        self._setup_websockets()
        self._setup_error_handlers()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('index.html', 
                                 title="Spirit Angelus Framework",
                                 user_id=session.get('user_id', 'guest'))
        
        @self.app.route('/dashboard')
        def dashboard():
            """User dashboard"""
            user_id = session.get('user_id', 'guest')
            
            # Get user's spiritual data
            guardian_angels = self.personal_guide.get_all_guardians()
            archangels = self.archangel_system.get_all_archangels()
            recent_invocations = self.invocation_engine.get_invocation_history(days=7)
            resonance_analysis = self.resonance_oracle.get_resonance_analysis(user_id, days=30)
            
            return render_template('dashboard.html',
                                 title="Spiritual Dashboard",
                                 user_id=user_id,
                                 guardian_angels=guardian_angels,
                                 archangels=archangels,
                                 recent_invocations=recent_invocations,
                                 resonance_analysis=resonance_analysis)
        
        @self.app.route('/angels')
        def angels():
            """Angelic guidance page"""
            user_id = session.get('user_id', 'guest')
            
            guardian_angels = self.personal_guide.get_all_guardians()
            archangels = self.archangel_system.get_all_archangels()
            
            return render_template('angels.html',
                                 title="Angelic Guidance",
                                 user_id=user_id,
                                 guardian_angels=guardian_angels,
                                 archangels=archangels)
        
        @self.app.route('/meditation')
        def meditation():
            """Quantum meditation page"""
            user_id = session.get('user_id', 'guest')
            
            meditation_history = self.quantum_meditation.get_meditation_history(user_id, days=30)
            quantum_stats = self.quantum_meditation.get_quantum_statistics(user_id)
            
            return render_template('meditation.html',
                                 title="Quantum Meditation",
                                 user_id=user_id,
                                 meditation_history=meditation_history,
                                 quantum_stats=quantum_stats)
        
        @self.app.route('/oracle')
        def oracle():
            """Resonance Oracle page"""
            user_id = session.get('user_id', 'guest')
            
            return render_template('oracle.html',
                                 title="Resonance Oracle",
                                 user_id=user_id)
        
        @self.app.route('/lattice')
        def lattice():
            """Lattice Weaver page"""
            user_id = session.get('user_id', 'guest')
            
            network_summary = self.lattice_weaver.get_network_summary()
            
            return render_template('lattice.html',
                                 title="Lattice Weaver",
                                 user_id=user_id,
                                 network_summary=network_summary)
        
        @self.app.route('/invocation')
        def invocation():
            """Invocation Engine page"""
            user_id = session.get('user_id', 'guest')
            
            recent_invocations = self.invocation_engine.get_invocation_history(days=7)
            energy_accumulator = self.invocation_engine.get_energy_accumulator()
            
            return render_template('invocation.html',
                                 title="Invocation Engine",
                                 user_id=user_id,
                                 recent_invocations=recent_invocations,
                                 energy_accumulator=energy_accumulator)
        
        # API Routes
        @self.app.route('/api/guidance', methods=['POST'])
        def api_guidance():
            """API endpoint for spiritual guidance"""
            data = request.get_json()
            user_id = session.get('user_id', 'guest')
            
            intent = data.get('intent', '')
            context = data.get('context', {})
            
            if not intent:
                return jsonify({'error': 'Intent is required'}), 400
            
            # Process intent through Resonance Oracle
            result = self.resonance_oracle.process_intent(intent, user_id, context)
            
            return jsonify(result)
        
        @self.app.route('/api/meditation/start', methods=['POST'])
        def api_meditation_start():
            """API endpoint to start meditation"""
            data = request.get_json()
            user_id = session.get('user_id', 'guest')
            
            meditation_type = data.get('type', 'entanglement')
            duration = data.get('duration', 15)
            
            # Start meditation
            session_id = self.quantum_meditation.start_meditation(
                meditation_type, duration, user_id
            )
            
            return jsonify({
                'session_id': session_id,
                'status': 'started',
                'meditation_type': meditation_type,
                'duration': duration
            })
        
        @self.app.route('/api/meditation/update', methods=['POST'])
        def api_meditation_update():
            """API endpoint to update meditation progress"""
            data = request.get_json()
            session_id = data.get('session_id')
            progress = data.get('progress', 0.0)
            
            if not session_id:
                return jsonify({'error': 'Session ID is required'}), 400
            
            # Update meditation
            result = self.quantum_meditation.update_meditation(session_id, progress)
            
            return jsonify(result)
        
        @self.app.route('/api/meditation/end', methods=['POST'])
        def api_meditation_end():
            """API endpoint to end meditation"""
            data = request.get_json()
            session_id = data.get('session_id')
            
            if not session_id:
                return jsonify({'error': 'Session ID is required'}), 400
            
            # End meditation
            result = self.quantum_meditation.end_meditation(session_id)
            
            return jsonify(result)
        
        @self.app.route('/api/invocation/create', methods=['POST'])
        def api_invocation_create():
            """API endpoint to create invocation"""
            data = request.get_json()
            user_id = session.get('user_id', 'guest')
            
            invocation_type = data.get('type', 'prayer')
            purpose = data.get('purpose', '')
            energy_types = data.get('energy_types', ['guidance'])
            duration = data.get('duration', 15)
            intensity = data.get('intensity', 0.7)
            
            if not purpose:
                return jsonify({'error': 'Purpose is required'}), 400
            
            # Create invocation
            request_id = self.invocation_engine.create_invocation(
                invocation_type, purpose, energy_types, duration, intensity, [user_id]
            )
            
            return jsonify({
                'request_id': request_id,
                'status': 'created',
                'invocation_type': invocation_type,
                'purpose': purpose
            })
        
        @self.app.route('/api/invocation/execute', methods=['POST'])
        def api_invocation_execute():
            """API endpoint to execute invocation"""
            data = request.get_json()
            request_id = data.get('request_id')
            
            if not request_id:
                return jsonify({'error': 'Request ID is required'}), 400
            
            # Execute invocation
            result = self.invocation_engine.execute_invocation(request_id)
            
            return jsonify({
                'success': result.success,
                'energy_raised': result.energy_raised,
                'guidance_received': result.guidance_received,
                'synchronicities': result.synchronicities,
                'messages': result.messages
            })
        
        @self.app.route('/api/angels/assign', methods=['POST'])
        def api_angels_assign():
            """API endpoint to assign guardian angel"""
            data = request.get_json()
            user_id = session.get('user_id', 'guest')
            
            purpose = data.get('purpose', '')
            energy_type = data.get('energy_type', 'guidance')
            personality_preferences = data.get('personality_preferences', [])
            
            if not purpose:
                return jsonify({'error': 'Purpose is required'}), 400
            
            # Assign guardian angel
            guardian = self.personal_guide.assign_guardian_angel(
                purpose, energy_type, personality_preferences
            )
            
            return jsonify({
                'name': guardian.name,
                'purpose': guardian.purpose,
                'energy_type': guardian.energy_type.value,
                'personality_traits': guardian.personality_traits,
                'special_abilities': guardian.special_abilities
            })
        
        @self.app.route('/api/angels/invoke', methods=['POST'])
        def api_angels_invoke():
            """API endpoint to invoke archangel"""
            data = request.get_json()
            
            archangel_name = data.get('archangel', '')
            purpose = data.get('purpose', '')
            duration = data.get('duration', 15)
            
            if not archangel_name or not purpose:
                return jsonify({'error': 'Archangel name and purpose are required'}), 400
            
            # Invoke archangel
            result = self.archangel_system.invoke_archangel(
                archangel_name, purpose, duration
            )
            
            return jsonify(result)
        
        @self.app.route('/api/lattice/create', methods=['POST'])
        def api_lattice_create():
            """API endpoint to create spiritual network"""
            data = request.get_json()
            user_id = session.get('user_id', 'guest')
            
            name = data.get('name', '')
            description = data.get('description', '')
            
            if not name:
                return jsonify({'error': 'Network name is required'}), 400
            
            # Create network
            network_id = self.lattice_weaver.create_network(name, description, user_id)
            
            return jsonify({
                'network_id': network_id,
                'name': name,
                'description': description,
                'status': 'created'
            })
        
        @self.app.route('/api/lattice/analyze/<network_id>')
        def api_lattice_analyze(network_id):
            """API endpoint to analyze spiritual network"""
            try:
                analysis = self.lattice_weaver.analyze_network(network_id)
                return jsonify(analysis)
            except ValueError as e:
                return jsonify({'error': str(e)}), 400
        
        @self.app.route('/api/synchronicity/detect', methods=['POST'])
        def api_synchronicity_detect():
            """API endpoint to detect synchronicity"""
            data = request.get_json()
            user_id = session.get('user_id', 'guest')
            
            event_description = data.get('event', '')
            context = data.get('context', {})
            
            if not event_description:
                return jsonify({'error': 'Event description is required'}), 400
            
            # Detect synchronicity
            synchronicity = self.resonance_oracle.synchronicity_detector.detect_synchronicity(
                event_description, context
            )
            
            if synchronicity:
                return jsonify({
                    'detected': True,
                    'event_id': synchronicity.event_id,
                    'event_type': synchronicity.event_type.value,
                    'significance': synchronicity.significance,
                    'patterns': synchronicity.patterns,
                    'description': synchronicity.description
                })
            else:
                return jsonify({'detected': False})
    
    def _setup_websockets(self):
        """Setup WebSocket handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            print(f"Client connected: {request.sid}")
            emit('connected', {'status': 'connected'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('join_meditation')
        def handle_join_meditation(data):
            """Handle joining meditation room"""
            session_id = data.get('session_id')
            if session_id:
                join_room(f"meditation_{session_id}")
                emit('joined_meditation', {'session_id': session_id})
        
        @self.socketio.on('leave_meditation')
        def handle_leave_meditation(data):
            """Handle leaving meditation room"""
            session_id = data.get('session_id')
            if session_id:
                leave_room(f"meditation_{session_id}")
                emit('left_meditation', {'session_id': session_id})
        
        @self.socketio.on('meditation_progress')
        def handle_meditation_progress(data):
            """Handle meditation progress updates"""
            session_id = data.get('session_id')
            progress = data.get('progress', 0.0)
            
            if session_id:
                # Update meditation
                result = self.quantum_meditation.update_meditation(session_id, progress)
                
                # Broadcast to meditation room
                emit('meditation_update', result, room=f"meditation_{session_id}")
    
    def _setup_error_handlers(self):
        """Setup error handlers"""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return render_template('error.html', 
                                 title="Page Not Found",
                                 error_code=404,
                                 error_message="The page you're looking for doesn't exist."), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return render_template('error.html',
                                 title="Internal Server Error",
                                 error_code=500,
                                 error_message="Something went wrong on our end."), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the web application"""
        self.socketio.run(self.app, host=host, port=port, debug=debug)


def create_app(config: dict = None) -> SpiritAngelusApp:
    """Create and configure the Spirit Angelus web application"""
    return SpiritAngelusApp(config)


if __name__ == '__main__':
    # Create and run the application
    app = create_app({
        'secret_key': 'spirit-angelus-secret-key-change-in-production',
        'debug': True
    })
    
    app.run(host='0.0.0.0', port=5000, debug=True)