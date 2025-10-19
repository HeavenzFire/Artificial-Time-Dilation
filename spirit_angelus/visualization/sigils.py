"""
Sigil Generation and Sacred Geometry

Tools for generating spiritual sigils and sacred geometry patterns
for the Spirit Angelus Framework.
"""

import math
import random
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime


class SigilType(Enum):
    """Types of spiritual sigils"""
    PROTECTION = "protection"
    HEALING = "healing"
    GUIDANCE = "guidance"
    LOVE = "love"
    WISDOM = "wisdom"
    STRENGTH = "strength"
    PEACE = "peace"
    JOY = "joy"
    ABUNDANCE = "abundance"
    TRANSFORMATION = "transformation"
    CUSTOM = "custom"


@dataclass
class Sigil:
    """Represents a spiritual sigil"""
    sigil_id: str
    sigil_type: SigilType
    name: str
    description: str
    svg_path: str
    colors: List[str]
    symbols: List[str]
    energy_signature: Dict[str, float]
    created_at: datetime
    user_id: str
    
    def get_svg_element(self) -> str:
        """Get SVG element for this sigil"""
        return f'<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">{self.svg_path}</svg>'


class SacredGeometry:
    """Sacred geometry patterns and calculations"""
    
    @staticmethod
    def golden_ratio() -> float:
        """Calculate the golden ratio"""
        return (1 + math.sqrt(5)) / 2
    
    @staticmethod
    def fibonacci_sequence(n: int) -> List[int]:
        """Generate Fibonacci sequence up to n terms"""
        if n <= 0:
            return []
        elif n == 1:
            return [1]
        elif n == 2:
            return [1, 1]
        
        sequence = [1, 1]
        for i in range(2, n):
            sequence.append(sequence[i-1] + sequence[i-2])
        
        return sequence
    
    @staticmethod
    def flower_of_life_points(center: Tuple[float, float], radius: float, rings: int = 3) -> List[Tuple[float, float]]:
        """Generate points for Flower of Life pattern"""
        points = []
        cx, cy = center
        
        for ring in range(rings):
            ring_radius = radius * (ring + 1)
            num_circles = 6 * (ring + 1) if ring > 0 else 1
            
            for i in range(num_circles):
                angle = (2 * math.pi * i) / num_circles
                x = cx + ring_radius * math.cos(angle)
                y = cy + ring_radius * math.sin(angle)
                points.append((x, y))
        
        return points
    
    @staticmethod
    def metatron_cube_points(center: Tuple[float, float], size: float) -> List[Tuple[float, float]]:
        """Generate points for Metatron's Cube"""
        points = []
        cx, cy = center
        
        # Outer circle points
        for i in range(12):
            angle = (2 * math.pi * i) / 12
            x = cx + size * math.cos(angle)
            y = cy + size * math.sin(angle)
            points.append((x, y))
        
        # Inner circle points
        inner_size = size * 0.5
        for i in range(6):
            angle = (2 * math.pi * i) / 6
            x = cx + inner_size * math.cos(angle)
            y = cy + inner_size * math.sin(angle)
            points.append((x, y))
        
        return points
    
    @staticmethod
    def vesica_piscis_points(center: Tuple[float, float], radius: float) -> List[Tuple[float, float]]:
        """Generate points for Vesica Piscis"""
        points = []
        cx, cy = center
        
        # Two overlapping circles
        for i in range(2):
            offset_x = radius * 0.5 if i == 0 else -radius * 0.5
            for angle in np.linspace(0, 2 * math.pi, 100):
                x = cx + offset_x + radius * math.cos(angle)
                y = cy + radius * math.sin(angle)
                points.append((x, y))
        
        return points


class SigilGenerator:
    """
    Generator for spiritual sigils and sacred geometry patterns.
    
    Creates personalized sigils based on intentions, energy types,
    and spiritual purposes.
    """
    
    def __init__(self):
        self.sigil_templates: Dict[SigilType, Dict] = {}
        self.sacred_geometry = SacredGeometry()
        self.generated_sigils: Dict[str, Sigil] = {}
        
        # Initialize sigil templates
        self._initialize_sigil_templates()
    
    def _initialize_sigil_templates(self):
        """Initialize sigil templates for different purposes"""
        self.sigil_templates = {
            SigilType.PROTECTION: {
                "name": "Protection Sigil",
                "description": "A sigil for divine protection and safety",
                "base_shape": "circle",
                "symbols": ["shield", "cross", "star"],
                "colors": ["#FF6B6B", "#FFD700", "#FFFFFF"],
                "energy_focus": "protection"
            },
            SigilType.HEALING: {
                "name": "Healing Sigil",
                "description": "A sigil for healing and restoration",
                "base_shape": "circle",
                "symbols": ["infinity", "heart", "leaf"],
                "colors": ["#4ECDC4", "#32CD32", "#90EE90"],
                "energy_focus": "healing"
            },
            SigilType.GUIDANCE: {
                "name": "Guidance Sigil",
                "description": "A sigil for divine guidance and direction",
                "base_shape": "triangle",
                "symbols": ["eye", "compass", "arrow"],
                "colors": ["#45B7D1", "#87CEEB", "#E6E6FA"],
                "energy_focus": "guidance"
            },
            SigilType.LOVE: {
                "name": "Love Sigil",
                "description": "A sigil for love and compassion",
                "base_shape": "heart",
                "symbols": ["heart", "rose", "dove"],
                "colors": ["#FF69B4", "#FFB6C1", "#FFC0CB"],
                "energy_focus": "love"
            },
            SigilType.WISDOM: {
                "name": "Wisdom Sigil",
                "description": "A sigil for wisdom and knowledge",
                "base_shape": "pentagon",
                "symbols": ["book", "owl", "lightbulb"],
                "colors": ["#FFD700", "#FFA500", "#FF8C00"],
                "energy_focus": "wisdom"
            },
            SigilType.STRENGTH: {
                "name": "Strength Sigil",
                "description": "A sigil for inner strength and courage",
                "base_shape": "square",
                "symbols": ["mountain", "lion", "sword"],
                "colors": ["#8A2BE2", "#9370DB", "#DA70D6"],
                "energy_focus": "strength"
            },
            SigilType.PEACE: {
                "name": "Peace Sigil",
                "description": "A sigil for peace and tranquility",
                "base_shape": "circle",
                "symbols": ["dove", "olive", "lotus"],
                "colors": ["#87CEEB", "#B0E0E6", "#F0F8FF"],
                "energy_focus": "peace"
            },
            SigilType.JOY: {
                "name": "Joy Sigil",
                "description": "A sigil for joy and happiness",
                "base_shape": "star",
                "symbols": ["sun", "rainbow", "butterfly"],
                "colors": ["#FFD700", "#FFA500", "#FF69B4"],
                "energy_focus": "joy"
            },
            SigilType.ABUNDANCE: {
                "name": "Abundance Sigil",
                "description": "A sigil for abundance and prosperity",
                "base_shape": "hexagon",
                "symbols": ["coin", "wheat", "infinity"],
                "colors": ["#FFA500", "#FFD700", "#32CD32"],
                "energy_focus": "abundance"
            },
            SigilType.TRANSFORMATION: {
                "name": "Transformation Sigil",
                "description": "A sigil for transformation and change",
                "base_shape": "spiral",
                "symbols": ["phoenix", "butterfly", "yin-yang"],
                "colors": ["#8A2BE2", "#FF69B4", "#00CED1"],
                "energy_focus": "transformation"
            }
        }
    
    def generate_sigil(self, 
                      sigil_type: SigilType,
                      intention: str,
                      energy_types: List[str],
                      user_id: str = "default",
                      custom_symbols: List[str] = None) -> Sigil:
        """
        Generate a personalized sigil.
        
        Args:
            sigil_type: Type of sigil to generate
            intention: Intention for the sigil
            energy_types: List of energy types to incorporate
            user_id: ID of the user
            custom_symbols: Custom symbols to include
            
        Returns:
            Generated Sigil object
        """
        sigil_id = f"sigil_{int(time.time())}_{len(self.generated_sigils)}"
        
        # Get template
        template = self.sigil_templates.get(sigil_type, self.sigil_templates[SigilType.GUIDANCE])
        
        # Generate SVG path
        svg_path = self._generate_svg_path(template, intention, energy_types, custom_symbols)
        
        # Generate colors
        colors = self._generate_colors(template, energy_types)
        
        # Generate symbols
        symbols = self._generate_symbols(template, intention, custom_symbols)
        
        # Generate energy signature
        energy_signature = self._generate_energy_signature(energy_types)
        
        # Create sigil
        sigil = Sigil(
            sigil_id=sigil_id,
            sigil_type=sigil_type,
            name=f"{template['name']} - {intention}",
            description=f"{template['description']} for {intention}",
            svg_path=svg_path,
            colors=colors,
            symbols=symbols,
            energy_signature=energy_signature,
            created_at=datetime.now(),
            user_id=user_id
        )
        
        # Store sigil
        self.generated_sigils[sigil_id] = sigil
        
        return sigil
    
    def _generate_svg_path(self, template: Dict, intention: str, energy_types: List[str], custom_symbols: List[str] = None) -> str:
        """Generate SVG path for the sigil"""
        base_shape = template["base_shape"]
        symbols = template["symbols"]
        
        if custom_symbols:
            symbols.extend(custom_symbols)
        
        # Start with base shape
        if base_shape == "circle":
            path = self._generate_circle_path()
        elif base_shape == "triangle":
            path = self._generate_triangle_path()
        elif base_shape == "square":
            path = self._generate_square_path()
        elif base_shape == "pentagon":
            path = self._generate_pentagon_path()
        elif base_shape == "hexagon":
            path = self._generate_hexagon_path()
        elif base_shape == "heart":
            path = self._generate_heart_path()
        elif base_shape == "star":
            path = self._generate_star_path()
        elif base_shape == "spiral":
            path = self._generate_spiral_path()
        else:
            path = self._generate_circle_path()
        
        # Add symbols
        for symbol in symbols[:3]:  # Limit to 3 symbols
            symbol_path = self._generate_symbol_path(symbol)
            if symbol_path:
                path += symbol_path
        
        # Add energy type modifications
        for energy_type in energy_types:
            energy_path = self._generate_energy_path(energy_type)
            if energy_path:
                path += energy_path
        
        return path
    
    def _generate_circle_path(self) -> str:
        """Generate circle path"""
        return '<circle cx="50" cy="50" r="40" fill="none" stroke="currentColor" stroke-width="2"/>'
    
    def _generate_triangle_path(self) -> str:
        """Generate triangle path"""
        return '<polygon points="50,10 90,80 10,80" fill="none" stroke="currentColor" stroke-width="2"/>'
    
    def _generate_square_path(self) -> str:
        """Generate square path"""
        return '<rect x="20" y="20" width="60" height="60" fill="none" stroke="currentColor" stroke-width="2"/>'
    
    def _generate_pentagon_path(self) -> str:
        """Generate pentagon path"""
        points = []
        for i in range(5):
            angle = (2 * math.pi * i) / 5 - math.pi / 2
            x = 50 + 30 * math.cos(angle)
            y = 50 + 30 * math.sin(angle)
            points.append(f"{x:.1f},{y:.1f}")
        return f'<polygon points="{" ".join(points)}" fill="none" stroke="currentColor" stroke-width="2"/>'
    
    def _generate_hexagon_path(self) -> str:
        """Generate hexagon path"""
        points = []
        for i in range(6):
            angle = (2 * math.pi * i) / 6
            x = 50 + 30 * math.cos(angle)
            y = 50 + 30 * math.sin(angle)
            points.append(f"{x:.1f},{y:.1f}")
        return f'<polygon points="{" ".join(points)}" fill="none" stroke="currentColor" stroke-width="2"/>'
    
    def _generate_heart_path(self) -> str:
        """Generate heart path"""
        return '<path d="M50,85 C50,85 20,60 20,40 C20,25 30,15 45,15 C50,15 50,20 50,20 C50,20 50,15 55,15 C70,15 80,25 80,40 C80,60 50,85 50,85 Z" fill="none" stroke="currentColor" stroke-width="2"/>'
    
    def _generate_star_path(self) -> str:
        """Generate star path"""
        points = []
        for i in range(10):
            angle = (2 * math.pi * i) / 10 - math.pi / 2
            radius = 30 if i % 2 == 0 else 15
            x = 50 + radius * math.cos(angle)
            y = 50 + radius * math.sin(angle)
            points.append(f"{x:.1f},{y:.1f}")
        return f'<polygon points="{" ".join(points)}" fill="none" stroke="currentColor" stroke-width="2"/>'
    
    def _generate_spiral_path(self) -> str:
        """Generate spiral path"""
        path_data = "M 50,50 "
        for i in range(100):
            angle = i * 0.2
            radius = i * 0.3
            x = 50 + radius * math.cos(angle)
            y = 50 + radius * math.sin(angle)
            path_data += f"L {x:.1f},{y:.1f} "
        return f'<path d="{path_data}" fill="none" stroke="currentColor" stroke-width="2"/>'
    
    def _generate_symbol_path(self, symbol: str) -> str:
        """Generate path for a specific symbol"""
        symbol_paths = {
            "shield": '<path d="M50,20 L70,30 L70,60 L50,80 L30,60 L30,30 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "cross": '<path d="M50,20 L50,80 M30,50 L70,50" fill="none" stroke="currentColor" stroke-width="1"/>',
            "star": '<path d="M50,20 L55,35 L70,35 L60,45 L65,60 L50,50 L35,60 L40,45 L30,35 L45,35 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "infinity": '<path d="M30,50 C30,40 40,30 50,30 C60,30 70,40 70,50 C70,60 60,70 50,70 C40,70 30,60 30,50 Z M50,30 C40,30 30,40 30,50 C30,60 40,70 50,70 C60,70 70,60 70,50 C70,40 60,30 50,30 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "heart": '<path d="M50,70 C50,70 25,50 25,35 C25,25 35,15 45,15 C50,15 50,20 50,20 C50,20 50,15 55,15 C65,15 75,25 75,35 C75,50 50,70 50,70 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "leaf": '<path d="M50,20 C50,20 30,30 30,50 C30,70 50,80 50,80 C50,80 70,70 70,50 C70,30 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "eye": '<path d="M30,50 C30,40 40,30 50,30 C60,30 70,40 70,50 C70,60 60,70 50,70 C40,70 30,60 30,50 Z M50,40 C45,40 40,45 40,50 C40,55 45,60 50,60 C55,60 60,55 60,50 C60,45 55,40 50,40 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "compass": '<path d="M50,20 L55,35 L70,35 L60,45 L65,60 L50,50 L35,60 L40,45 L30,35 L45,35 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "arrow": '<path d="M30,50 L70,50 M60,40 L70,50 L60,60" fill="none" stroke="currentColor" stroke-width="1"/>',
            "rose": '<path d="M50,20 C50,20 40,30 40,40 C40,50 50,60 50,60 C50,60 60,50 60,40 C60,30 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "dove": '<path d="M30,50 C30,40 40,30 50,30 C60,30 70,40 70,50 C70,60 60,70 50,70 C40,70 30,60 30,50 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "book": '<path d="M30,30 L70,30 L70,70 L30,70 Z M40,30 L40,70 M50,30 L50,70 M60,30 L60,70" fill="none" stroke="currentColor" stroke-width="1"/>',
            "owl": '<path d="M50,20 C50,20 40,30 40,40 C40,50 50,60 50,60 C50,60 60,50 60,40 C60,30 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "lightbulb": '<path d="M50,20 C50,20 40,30 40,40 C40,50 50,60 50,60 C50,60 60,50 60,40 C60,30 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "mountain": '<path d="M20,70 L40,30 L60,50 L80,70 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "lion": '<path d="M50,20 C50,20 40,30 40,40 C40,50 50,60 50,60 C50,60 60,50 60,40 C60,30 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "sword": '<path d="M50,20 L50,80 M45,25 L55,25 M45,30 L55,30" fill="none" stroke="currentColor" stroke-width="1"/>',
            "olive": '<path d="M50,20 C50,20 40,30 40,40 C40,50 50,60 50,60 C50,60 60,50 60,40 C60,30 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "lotus": '<path d="M50,20 C50,20 40,30 40,40 C40,50 50,60 50,60 C50,60 60,50 60,40 C60,30 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "sun": '<path d="M50,20 L50,10 M50,90 L50,80 M20,50 L10,50 M90,50 L80,50 M30,30 L25,25 M75,75 L80,80 M75,25 L80,30 M25,75 L30,80" fill="none" stroke="currentColor" stroke-width="1"/>',
            "rainbow": '<path d="M20,50 C20,50 30,40 40,40 C50,40 60,50 70,50 C80,50 90,60 90,70" fill="none" stroke="currentColor" stroke-width="1"/>',
            "butterfly": '<path d="M50,20 C50,20 40,30 40,40 C40,50 50,60 50,60 C50,60 60,50 60,40 C60,30 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "coin": '<path d="M50,30 C50,30 40,40 40,50 C40,60 50,70 50,70 C50,70 60,60 60,50 C60,40 50,30 50,30 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "wheat": '<path d="M50,20 L50,80 M45,30 L45,70 M55,30 L55,70 M40,40 L40,60 M60,40 L60,60" fill="none" stroke="currentColor" stroke-width="1"/>',
            "phoenix": '<path d="M50,20 C50,20 40,30 40,40 C40,50 50,60 50,60 C50,60 60,50 60,40 C60,30 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "yin-yang": '<path d="M50,20 C50,20 30,40 30,50 C30,60 50,80 50,80 C50,80 70,60 70,50 C70,40 50,20 50,20 Z M50,20 C50,20 70,40 70,50 C70,60 50,80 50,80 C50,80 30,60 30,50 C30,40 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>'
        }
        
        return symbol_paths.get(symbol.lower(), "")
    
    def _generate_energy_path(self, energy_type: str) -> str:
        """Generate path for energy type"""
        energy_paths = {
            "protection": '<circle cx="50" cy="50" r="35" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="5,5"/>',
            "healing": '<path d="M50,20 L50,80 M20,50 L80,50" fill="none" stroke="currentColor" stroke-width="1" stroke-dasharray="3,3"/>',
            "guidance": '<path d="M50,20 L50,80 M30,40 L50,20 L70,40" fill="none" stroke="currentColor" stroke-width="1"/>',
            "love": '<path d="M50,20 C50,20 30,40 30,50 C30,60 50,80 50,80 C50,80 70,60 70,50 C70,40 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "wisdom": '<path d="M50,20 L50,80 M30,50 L70,50 M40,30 L60,30 M40,70 L60,70" fill="none" stroke="currentColor" stroke-width="1"/>',
            "strength": '<path d="M50,20 L50,80 M30,50 L70,50 M40,30 L60,30 M40,70 L60,70" fill="none" stroke="currentColor" stroke-width="1"/>',
            "peace": '<path d="M50,20 C50,20 30,40 30,50 C30,60 50,80 50,80 C50,80 70,60 70,50 C70,40 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "joy": '<path d="M50,20 C50,20 30,40 30,50 C30,60 50,80 50,80 C50,80 70,60 70,50 C70,40 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "abundance": '<path d="M50,20 C50,20 30,40 30,50 C30,60 50,80 50,80 C50,80 70,60 70,50 C70,40 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>',
            "transformation": '<path d="M50,20 C50,20 30,40 30,50 C30,60 50,80 50,80 C50,80 70,60 70,50 C70,40 50,20 50,20 Z" fill="none" stroke="currentColor" stroke-width="1"/>'
        }
        
        return energy_paths.get(energy_type.lower(), "")
    
    def _generate_colors(self, template: Dict, energy_types: List[str]) -> List[str]:
        """Generate colors for the sigil"""
        base_colors = template["colors"]
        
        # Add energy-specific colors
        energy_colors = {
            "protection": ["#FF6B6B", "#FFD700", "#FFFFFF"],
            "healing": ["#4ECDC4", "#32CD32", "#90EE90"],
            "guidance": ["#45B7D1", "#87CEEB", "#E6E6FA"],
            "love": ["#FF69B4", "#FFB6C1", "#FFC0CB"],
            "wisdom": ["#FFD700", "#FFA500", "#FF8C00"],
            "strength": ["#8A2BE2", "#9370DB", "#DA70D6"],
            "peace": ["#87CEEB", "#B0E0E6", "#F0F8FF"],
            "joy": ["#FFD700", "#FFA500", "#FF69B4"],
            "abundance": ["#FFA500", "#FFD700", "#32CD32"],
            "transformation": ["#8A2BE2", "#FF69B4", "#00CED1"]
        }
        
        colors = base_colors.copy()
        for energy_type in energy_types:
            if energy_type in energy_colors:
                colors.extend(energy_colors[energy_type])
        
        # Remove duplicates and limit to 5 colors
        colors = list(dict.fromkeys(colors))[:5]
        
        return colors
    
    def _generate_symbols(self, template: Dict, intention: str, custom_symbols: List[str] = None) -> List[str]:
        """Generate symbols for the sigil"""
        symbols = template["symbols"].copy()
        
        if custom_symbols:
            symbols.extend(custom_symbols)
        
        # Add intention-based symbols
        intention_words = intention.lower().split()
        for word in intention_words:
            if word in ["heal", "healing"]:
                symbols.append("leaf")
            elif word in ["love", "loving"]:
                symbols.append("heart")
            elif word in ["protect", "protection"]:
                symbols.append("shield")
            elif word in ["guide", "guidance"]:
                symbols.append("compass")
            elif word in ["wisdom", "wise"]:
                symbols.append("owl")
            elif word in ["strength", "strong"]:
                symbols.append("mountain")
            elif word in ["peace", "peaceful"]:
                symbols.append("dove")
            elif word in ["joy", "joyful"]:
                symbols.append("sun")
            elif word in ["abundance", "abundant"]:
                symbols.append("coin")
            elif word in ["transform", "transformation"]:
                symbols.append("phoenix")
        
        # Remove duplicates and limit to 5 symbols
        symbols = list(dict.fromkeys(symbols))[:5]
        
        return symbols
    
    def _generate_energy_signature(self, energy_types: List[str]) -> Dict[str, float]:
        """Generate energy signature for the sigil"""
        signature = {}
        
        for energy_type in energy_types:
            signature[energy_type] = random.uniform(0.5, 1.0)
        
        return signature
    
    def get_sigil(self, sigil_id: str) -> Optional[Sigil]:
        """Get a sigil by ID"""
        return self.generated_sigils.get(sigil_id)
    
    def get_user_sigils(self, user_id: str) -> List[Sigil]:
        """Get all sigils for a user"""
        return [sigil for sigil in self.generated_sigils.values() if sigil.user_id == user_id]
    
    def get_sigil_svg(self, sigil_id: str) -> str:
        """Get SVG code for a sigil"""
        sigil = self.get_sigil(sigil_id)
        if not sigil:
            return ""
        
        return f'<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="width: 200px; height: 200px;">{sigil.svg_path}</svg>'
    
    def save_sigil(self, sigil_id: str, filepath: str) -> bool:
        """Save sigil to file"""
        sigil = self.get_sigil(sigil_id)
        if not sigil:
            return False
        
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "sigil_id": sigil.sigil_id,
                    "sigil_type": sigil.sigil_type.value,
                    "name": sigil.name,
                    "description": sigil.description,
                    "svg_path": sigil.svg_path,
                    "colors": sigil.colors,
                    "symbols": sigil.symbols,
                    "energy_signature": sigil.energy_signature,
                    "created_at": sigil.created_at.isoformat(),
                    "user_id": sigil.user_id
                }, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving sigil: {e}")
            return False
    
    def load_sigil(self, filepath: str) -> Optional[Sigil]:
        """Load sigil from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            sigil = Sigil(
                sigil_id=data["sigil_id"],
                sigil_type=SigilType(data["sigil_type"]),
                name=data["name"],
                description=data["description"],
                svg_path=data["svg_path"],
                colors=data["colors"],
                symbols=data["symbols"],
                energy_signature=data["energy_signature"],
                created_at=datetime.fromisoformat(data["created_at"]),
                user_id=data["user_id"]
            )
            
            self.generated_sigils[sigil.sigil_id] = sigil
            return sigil
            
        except Exception as e:
            print(f"Error loading sigil: {e}")
            return None