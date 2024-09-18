#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Byzantium Attack Emulator - Enhanced with PyQt5 for Interactivity,
Internationalization support using YAML language files.
"""

import sys
import os
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QInputDialog, QSlider, QHBoxLayout, QProgressBar, QTextEdit, QGroupBox, QMessageBox, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon, QImage, QPainter
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import yaml
import pathlib

# Ensure PyYAML is installed. If not, instruct the user to install it.
try:
    import yaml
except ImportError:
    print("PyYAML is not installed. Please install it using 'pip install pyyaml'")
    sys.exit(1)


class Translator:
    """Handles loading and retrieving languages from YAML files."""

    def __init__(self, languages_dir='languages', config_file='config.yaml'):
        self.languages_dir = languages_dir
        self.config_file = config_file
        self.translations = {}
        self.current_language = 'en'  # Default language

        self.load_config()
        self.load_language()

    def load_config(self):
        """Load the language selection from the config file."""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                try:
                    config = yaml.safe_load(f)
                    if config and 'language' in config:
                        self.current_language = config['language']
                except yaml.YAMLError:
                    print("Error parsing config.yaml. Using default language 'en'.")
        else:
            # Config does not exist; will prompt user later
            pass

    def save_config(self):
        """Save the current language selection to the config file."""
        config = {'language': self.current_language}
        with open(self.config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

    def get_available_languages(self):
        """List available language YAML files."""
        if not os.path.exists(self.languages_dir):
            print(f"Languages directory '{self.languages_dir}' not found.")
            return []

        files = [f for f in os.listdir(self.languages_dir) if f.endswith('.yaml')]
        languages = [os.path.splitext(f)[0] for f in files]
        return languages

    def select_language(self):
        """Prompt the user to select a language."""
        available_languages = self.get_available_languages()
        if not available_languages:
            print("No language files found. Ensure that the 'languages' directory contains YAML files.")
            sys.exit(1)

        language_names = {
            'en': 'English',
            'it': 'Italiano',
            'fr': 'Français',
            'es': 'Espagnol'
            # Add other language codes and names as needed
        }

        # Create a list of language display names
        language_display = [f"{code} - {language_names.get(code, code)}" for code in available_languages]

        # Prompt the user to select a language
        lang, ok = QInputDialog.getItem(
            None, "Select Language", "Choose your language:", language_display, 0, False
        )

        if ok and lang:
            # Extract language code
            selected_code = lang.split(' - ')[0]
            self.current_language = selected_code
            self.save_config()
        else:
            # User cancelled; proceed with default language
            pass

    def load_language(self):
        """Load languages from the selected language YAML file."""
        lang_file = os.path.join(self.languages_dir, f"{self.current_language}.yaml")
        if not os.path.exists(lang_file):
            QMessageBox.critical(
                None, "Language File Missing",
                f"Language file '{self.current_language}.yaml' not found in '{self.languages_dir}' directory."
            )
            sys.exit(1)

        with open(lang_file, 'r', encoding='utf-8') as f:
            try:
                self.translations = yaml.safe_load(f)
            except yaml.YAMLError as e:
                QMessageBox.critical(
                    None, "YAML Parsing Error",
                    f"Error parsing language file '{self.current_language}.yaml'.\n{str(e)}"
                )
                sys.exit(1)

    def tr(self, key, **kwargs):
        """Retrieve the translated string for the given key, formatted with kwargs."""
        text = self.translations.get(key, key)  # Fallback to key if not found
        try:
            return text.format(**kwargs)
        except KeyError as e:
            print(f"Missing placeholder in translation for key '{key}': {e}")
            return text  # Return unformatted text


class ByzantiumEmulator(QMainWindow):
    def __init__(self, translator, n_nodes=12):
        super().__init__()
        self.translator = translator  # Translator instance for i18n
        self.setWindowTitle(self.translator.tr('window_title'))
        self.setGeometry(100, 100, 1600, 900)  # Increased width to accommodate console and legend

        # Load images
        try:
            self.pirate_img = mpimg.imread(os.path.join('images', 'pirate.png'))
            self.byzantium_img = mpimg.imread(os.path.join('images', 'honest.png'))
            self.confused_img = mpimg.imread(os.path.join('images', 'confused.png'))
            self.blacklisted_img = mpimg.imread(os.path.join('images', 'blacklisted.png'))
        except FileNotFoundError as e:
            QMessageBox.critical(
                self, self.translator.tr('image_not_found'),
                self.translator.tr('image_not_found_message', filename=e.filename)
            )
            sys.exit(1)

        # Initialize simulation parameters
        self.n_nodes = n_nodes
        self.phase_counter = 1  # Renamed from self.l to phase_counter for clarity
        self.TG_sub_nodes = []  # List of compromised nodes
        self.detected_malicious = set()  # Set of detected malicious nodes
        self.message_data = []  # To store tuples of (sender, receiver, message, color)
        self.messages = []  # To store FancyArrowPatch objects
        self.msg_animation = None
        self.msg_animation_index = 0
        self.messages_per_step = 10  # Increased from 5 to 10 for better detection

        # Initialize thresholds with default values
        self.detection_threshold = 0.6  # Default 60%
        self.system_compromise_threshold = 0.5  # Set to 50% to align with Bitcoin's 51% concept

        self.bft_threshold = (n_nodes - 1) // 3  # BFT fault tolerance threshold (~33%)

        # Initialize simulation speed (delay in milliseconds)
        self.simulation_speed = 100  # Default delay of 100 ms

        self.initialize_graph()

        # Set up the matplotlib Figure and Canvas
        self.figure, self.ax = plt.subplots(figsize=(12, 12))
        self.canvas = FigureCanvas(self.figure)

        # Set up UI Elements
        self.next_button = QPushButton(self.translator.tr('next_phase_button'))
        self.next_button.clicked.connect(self.next_phase)

        # Info Label
        self.info_label = QLabel(self.translator.tr('welcome_message'))
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setWordWrap(True)

        # Sliders for Configurable Thresholds and Simulation Speed
        self.setup_threshold_sliders()

        # Slider for Messages Per Phase
        self.slider_label = QLabel(self.translator.tr('messages_per_phase_label', value=self.messages_per_step))
        self.slider_label.setAlignment(Qt.AlignCenter)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(20)
        self.slider.setValue(self.messages_per_step)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.update_messages_per_phase)

        # Slider for Simulation Speed
        self.speed_label = QLabel(self.translator.tr('simulation_speed_label', value=self.simulation_speed))
        self.speed_label.setAlignment(Qt.AlignCenter)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)  # Slider range from 1 to 10
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(2)  # Default value as per current speed
        self.speed_slider.setTickInterval(1)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.valueChanged.connect(self.update_simulation_speed)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(self.translator.tr('phase_progress_format'))
        self.progress_bar.setAlignment(Qt.AlignCenter)

        # Side Console
        self.setup_side_console()

        # Legend
        self.setup_legend()

        # Layouts
        main_layout = QHBoxLayout()

        # Left Side: Graph and Controls
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.canvas)
        left_layout.addWidget(self.info_label)

        # Group Box for Sliders
        sliders_group = QGroupBox(self.translator.tr('configuration_sliders'))
        sliders_layout = QVBoxLayout()
        sliders_layout.addWidget(self.detection_threshold_label)
        sliders_layout.addWidget(self.detection_threshold_slider)
        sliders_layout.addWidget(self.system_compromise_label)
        sliders_layout.addWidget(self.system_compromise_slider)
        sliders_layout.addWidget(self.slider_label)
        sliders_layout.addWidget(self.slider)
        sliders_layout.addWidget(self.speed_label)
        sliders_layout.addWidget(self.speed_slider)
        sliders_group.setLayout(sliders_layout)

        left_layout.addWidget(sliders_group)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.next_button)

        # Right Side: Side Console and Legend
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel(self.translator.tr('console_output')))
        right_layout.addWidget(self.console)
        right_layout.addWidget(QLabel(self.translator.tr('legend')))
        right_layout.addWidget(self.legend_group)

        # Add to Main Layout
        main_layout.addLayout(left_layout, 3)  # 3/4 of the space
        main_layout.addLayout(right_layout, 1)  # 1/4 of the space

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Initial Plot
        self.plot_graph(initial=True)

    def setup_threshold_sliders(self):
        """Set up sliders for configurable thresholds."""
        # Detection Threshold Slider
        self.detection_threshold_label = QLabel(
            self.translator.tr('detection_threshold_label', value=int(self.detection_threshold * 100))
        )
        self.detection_threshold_label.setAlignment(Qt.AlignCenter)
        self.detection_threshold_slider = QSlider(Qt.Horizontal)
        self.detection_threshold_slider.setMinimum(30)  # Represents 30%
        self.detection_threshold_slider.setMaximum(80)  # Represents 80%
        self.detection_threshold_slider.setValue(int(self.detection_threshold * 100))
        self.detection_threshold_slider.setTickInterval(5)
        self.detection_threshold_slider.setTickPosition(QSlider.TicksBelow)
        self.detection_threshold_slider.valueChanged.connect(self.update_detection_threshold)

        # System Compromise Threshold Slider
        self.system_compromise_label = QLabel(
            self.translator.tr('system_compromise_threshold_label', value=int(self.system_compromise_threshold * 100))
        )
        self.system_compromise_label.setAlignment(Qt.AlignCenter)
        self.system_compromise_slider = QSlider(Qt.Horizontal)
        self.system_compromise_slider.setMinimum(30)  # Represents 30%
        self.system_compromise_slider.setMaximum(70)  # Represents 70%
        self.system_compromise_slider.setValue(int(self.system_compromise_threshold * 100))
        self.system_compromise_slider.setTickInterval(5)
        self.system_compromise_slider.setTickPosition(QSlider.TicksBelow)
        self.system_compromise_slider.valueChanged.connect(self.update_system_compromise_threshold)

    def setup_side_console(self):
        """Set up the side console for logging messages."""
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Courier", 10))
        self.console.setStyleSheet("background-color: #f0f0f0;")
        self.console.setMinimumWidth(300)

    def setup_legend(self):
        """Set up the legend for node types and message types."""
        self.legend_group = QGroupBox()
        legend_layout = QGridLayout()

        # Node Legends
        node_legends = [
            ("honest_node_legend", "honest.png"),
            ("malicious_node_legend", "pirate.png"),
            ("blacklisted_node_legend", "blacklisted.png"),
            ("confused_node_legend", "confused.png")
        ]

        for i, (label_key, img_filename) in enumerate(node_legends):
            img_path = os.path.join('images', img_filename)
            if not os.path.exists(img_path):
                QMessageBox.warning(
                    self, self.translator.tr('image_missing_title'),
                    self.translator.tr('image_missing_message', img_path=img_path)
                )
                # Create a blank pixmap if image is missing to avoid crashes
                pixmap = QPixmap(40, 40)
                pixmap.fill(Qt.transparent)
            else:
                pixmap = QPixmap(img_path)
                pixmap = pixmap.scaled(40, 40, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            node_label = QLabel(self.translator.tr(label_key))
            node_image = QLabel()
            node_image.setPixmap(pixmap)
            legend_layout.addWidget(node_image, i, 0)
            legend_layout.addWidget(node_label, i, 1)

        # Arrow Legends
        arrow_legends = [
            ("honest_message_legend", "green"),
            ("malicious_message_legend", "red")
        ]

        for j, (label_key, color) in enumerate(arrow_legends):
            arrow_label = QLabel(self.translator.tr(label_key))
            arrow = FancyArrowPatch(
                (0, 0), (1, 0),
                arrowstyle='->', mutation_scale=15,
                color=color, linewidth=2
            )
            arrow_fig, arrow_ax = plt.subplots(figsize=(1, 0.2))
            arrow_ax.add_patch(arrow)
            arrow_ax.set_xlim(-0.1, 1.1)
            arrow_ax.set_ylim(-0.1, 0.1)
            arrow_ax.axis('off')
            temp_arrow_path = 'temp_arrow.png'
            plt.savefig(temp_arrow_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(arrow_fig)
            if not os.path.exists(temp_arrow_path):
                QMessageBox.warning(
                    self, self.translator.tr('arrow_image_missing_title'),
                    self.translator.tr('arrow_image_missing_message')
                )
                arrow_pixmap = QPixmap(100, 20)
                arrow_pixmap.fill(Qt.transparent)
            else:
                arrow_pixmap = QPixmap(temp_arrow_path)
                arrow_pixmap = arrow_pixmap.scaled(100, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                os.remove(temp_arrow_path)  # Remove the temporary file

            arrow_image = QLabel()
            arrow_image.setPixmap(arrow_pixmap)
            legend_layout.addWidget(arrow_image, j + len(node_legends), 0)
            legend_layout.addWidget(arrow_label, j + len(node_legends), 1)

        self.legend_group.setLayout(legend_layout)

    def log_message(self, message, msg_type='info'):
        """Log messages to the side console with formatting based on message type."""
        if msg_type == 'info':
            formatted_message = f"{message}"
        elif msg_type == 'warning':
            formatted_message = f"<b>{message}</b>"
        elif msg_type == 'critical':
            formatted_message = f"<b><font color='red'>{message}</font></b>"
        else:
            formatted_message = f"{message}"

        # Replace any unintended HTML tags to prevent display issues
        # For example, escape any '<' or '>' not part of HTML formatting
        # This can be enhanced as needed
        formatted_message = formatted_message.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Re-introduce necessary HTML formatting for known tags
        formatted_message = formatted_message.replace('&lt;br&gt;', '<br>')
        formatted_message = formatted_message.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
        formatted_message = formatted_message.replace('&lt;font color=\'red\'&gt;', '<font color="red">').replace(
            '&lt;/font&gt;', '</font>')

        self.console.append(formatted_message)

    def update_detection_threshold(self, value):
        """Update the detection threshold based on slider."""
        self.detection_threshold = value / 100.0
        self.detection_threshold_label.setText(
            self.translator.tr('detection_threshold_label', value=value)
        )
        # Removed console logging for parameter change

    def update_system_compromise_threshold(self, value):
        """Update the system compromise threshold based on slider."""
        self.system_compromise_threshold = value / 100.0
        self.system_compromise_label.setText(
            self.translator.tr('system_compromise_threshold_label', value=value)
        )
        # Recalculate BFT threshold based on new total nodes
        self.recalculate_thresholds()

    def update_simulation_speed(self, value):
        """Update the simulation speed based on slider value."""
        # Mapping slider value (1-10) to delay (200 ms / value)
        self.simulation_speed = int(200 / value)
        self.speed_label.setText(
            self.translator.tr('simulation_speed_label', value=self.simulation_speed)
        )
        # No console logging needed as per user request

    def initialize_graph(self):
        """Initialize the graph with all loyal nodes."""
        self.TG = nx.Graph()
        for i in range(1, self.n_nodes + 1):
            self.TG.add_node(i, image=self.byzantium_img, size=0.03, status='honest')
        # Create a complete graph
        self.TG.add_edges_from([
            (i, j) for i in self.TG.nodes() for j in self.TG.nodes() if i < j
        ])
        self.POS = nx.spring_layout(self.TG, seed=42)  # Fixed layout for consistency

    def recalculate_thresholds(self):
        """Recalculate thresholds based on the current number of active nodes."""
        active_nodes = self.get_active_nodes()
        if len(active_nodes) > 0:
            self.bft_threshold = (len(active_nodes) - 1) // 3
            self.log_message(
                self.translator.tr(
                    'updated_bft_threshold',
                    threshold=self.bft_threshold,
                    active_nodes=len(active_nodes)
                ),
                'info'
            )
        else:
            self.bft_threshold = 0
            self.log_message(
                self.translator.tr('no_active_nodes_threshold'),
                'warning'
            )

    def get_active_nodes(self):
        """Get the list of active nodes (excluding blacklisted nodes)."""
        return [n for n in self.TG.nodes() if self.TG.nodes[n]['status'] != 'blacklisted']

    def plot_graph(self, initial=False):
        """Plot the current state of the graph."""
        self.ax.clear()
        self.ax.set_title(self.translator.tr('graph_title'), fontsize=18)
        self.ax.axis('off')

        # Draw edges
        nx.draw_networkx_edges(
            self.TG, pos=self.POS, edge_color="lightgray", ax=self.ax
        )

        # Draw nodes with images
        for n in self.TG.nodes():
            x, y = self.POS[n]
            image = self.TG.nodes[n]['image']
            size = self.TG.nodes[n]['size']
            ab = AnnotationBbox(
                OffsetImage(image, zoom=size), (x, y), frameon=False, zorder=1
            )
            self.ax.add_artist(ab)
            # Add node labels
            self.ax.text(x, y - 0.05, str(n), horizontalalignment='center', fontsize=8, zorder=4)

        # Draw existing messages
        for arrow in self.messages:
            self.ax.add_patch(arrow)

        # Draw detected malicious nodes with a red cross
        for n in self.detected_malicious:
            x, y = self.POS[n]
            # Draw a red cross
            cross_size = 0.05
            line1 = Line2D([x - cross_size, x + cross_size], [y - cross_size, y + cross_size],
                           color='red', linewidth=2, zorder=5)
            line2 = Line2D([x - cross_size, x + cross_size], [y + cross_size, y - cross_size],
                           color='red', linewidth=2, zorder=5)
            self.ax.add_line(line1)
            self.ax.add_line(line2)

        # Draw blacklisted nodes with blacklisted.png and grey circle overlay
        for n in self.TG.nodes():
            if self.TG.nodes[n]['status'] == 'blacklisted':
                x, y = self.POS[n]
                # Overlay a grey circle to signify blacklisting
                circ = plt.Circle((x, y), 0.07, color='grey', fill=True, alpha=0.5, zorder=4)
                self.ax.add_patch(circ)

        self.canvas.draw()

    def update_info(self, text):
        """Update the information label."""
        # Replace any remaining <br> with proper newlines if necessary
        sanitized_text = text.replace("<br>", "\n")
        self.info_label.setText(sanitized_text)

    def update_messages_per_phase(self):
        """Update the number of messages per phase based on slider value."""
        self.messages_per_step = self.slider.value()
        self.slider_label.setText(
            self.translator.tr('messages_per_phase_label', value=self.messages_per_step)
        )
        # Optional: Log message to console if desired
        # self.log_message(f"Messages per Phase set to {self.messages_per_step}.", 'info')

    def next_phase(self):
        """Proceed to the next phase in the simulation."""
        self.messages = []  # Clear messages from previous phases
        self.plot_graph()  # Refresh graph to remove old arrows

        # Blacklist previously detected malicious nodes
        nodes_to_blacklist = list(self.detected_malicious)
        for n in nodes_to_blacklist:
            self.TG.nodes[n]['status'] = 'blacklisted'
            self.TG.nodes[n]['image'] = self.blacklisted_img  # Change image to blacklisted.png
        if nodes_to_blacklist:
            node_list = ', '.join(map(str, nodes_to_blacklist))
            self.log_message(
                self.translator.tr('nodes_blacklisted', nodes=node_list),
                'critical'
            )
            QMessageBox.information(
                self, self.translator.tr('nodes_blacklisted_title'),
                self.translator.tr('nodes_blacklisted', nodes=node_list)
            )
            self.detected_malicious.clear()  # Clear the set after blacklisting
            self.recalculate_thresholds()

        if self.phase_counter == 1:
            # Initial Phase Explanation
            explanation = self.translator.tr('phase_info_initial')
            self.update_info(explanation)
            self.log_message(
                self.translator.tr('phase_log_initial'),
                'info'
            )
            self.phase_counter += 1
            return

        if self.phase_counter <= self.n_nodes:
            # Attack Phase
            available_nodes = list(set(self.TG.nodes()) - set(self.TG_sub_nodes) - set(
                [n for n in self.TG.nodes() if self.TG.nodes[n]['status'] == 'blacklisted']))
            if not available_nodes:
                self.log_message(
                    self.translator.tr('all_nodes_compromised'),
                    'critical'
                )
                QMessageBox.critical(
                    self, self.translator.tr('simulation_complete_title'),
                    self.translator.tr('final_state_all_compromised')
                )
                self.update_info(
                    self.translator.tr('final_state_all_compromised')
                )
                self.next_button.setEnabled(False)
                return

            # Compromise a random node
            N = random.choice(available_nodes)
            self.TG_sub_nodes.append(N)
            self.TG.nodes[N]['image'] = self.pirate_img  # Change image to pirate.png
            self.TG.nodes[N]['size'] = 0.03  # Adjust size to match other nodes
            self.TG.nodes[N]['status'] = 'malicious'
            self.plot_graph()
            self.log_message(
                self.translator.tr('phase_node_compromised', phase=self.phase_counter, node=N),
                'warning'
            )

            # Check if malicious nodes exceed System Compromise Threshold
            malicious_nodes = [node for node in self.TG.nodes() if self.TG.nodes[node]['status'] == 'malicious']
            active_nodes = self.get_active_nodes()
            proportion_malicious = len(malicious_nodes) / len(active_nodes) if active_nodes else 0

            if proportion_malicious > self.system_compromise_threshold:
                # System is compromised
                self.system_compromised = True
                self.log_message(
                    self.translator.tr('system_compromised'),
                    'critical'
                )
                QMessageBox.critical(
                    self, self.translator.tr('system_compromised_title'),
                    self.translator.tr('system_compromised_message')
                )
            else:
                self.system_compromised = False

            # Send messages from generals
            self.send_messages()

            # Update explanations
            compromised = len(malicious_nodes)
            proportion = proportion_malicious

            if compromised >= len(active_nodes) // 2:
                comment = self.translator.tr('warning_majority')
            elif compromised == len(active_nodes) - 1:
                comment = self.translator.tr('base_ideology_changed')
            elif compromised >= (2 * len(active_nodes)) / 3:
                comment = self.translator.tr('majority_compromised')
            elif compromised >= len(active_nodes) / 3:
                comment = self.translator.tr('significant_compromised')
            elif compromised >= len(active_nodes) / 4:
                comment = self.translator.tr('initial_phase_compromise')
            else:
                comment = self.translator.tr('stable_system')

            info_text = self.translator.tr(
                'non_loyal_part',
                compromised=compromised,
                total=len(active_nodes),
                proportion=proportion
            ) + "\n" + comment
            self.update_info(info_text)
            self.log_message(
                self.translator.tr('phase_info', info=info_text),
                'info'
            )

            # Provide explanations
            if compromised == 1:
                phase_explanation = self.translator.tr('stage_1_explanation')
                self.log_message(phase_explanation, 'info')
            elif compromised == len(active_nodes) // 4:
                phase_explanation = self.translator.tr('stage_2_explanation')
                self.log_message(phase_explanation, 'info')
            elif compromised == len(active_nodes) // 3:
                phase_explanation = self.translator.tr('stage_3_explanation')
                self.log_message(phase_explanation, 'info')
            elif compromised >= (2 * len(active_nodes)) // 3:
                phase_explanation = self.translator.tr('stage_4_explanation')
                self.log_message(phase_explanation, 'critical')
                QMessageBox.warning(
                    self, self.translator.tr('majority_compromised_title'),
                    self.translator.tr('majority_compromised_message')
                )
                bft_explanation = self.translator.tr('bft_explanation')
                self.log_message(bft_explanation, 'critical')
            elif compromised >= len(active_nodes) // 2:
                phase_explanation = self.translator.tr('critical_stage_explanation')
                self.log_message(phase_explanation, 'critical')
                QMessageBox.critical(
                    self, self.translator.tr('critical_stage_title'),
                    self.translator.tr('critical_stage_message')
                )
                failure_explanation = self.translator.tr('failure_explanation')
                self.log_message(failure_explanation, 'critical')
                self.convert_honest_nodes()
            elif compromised == len(active_nodes) - 1:
                phase_explanation = self.translator.tr('final_stage_explanation')
                self.log_message(phase_explanation, 'critical')
                QMessageBox.critical(
                    self, self.translator.tr('system_collapse_title'),
                    self.translator.tr('system_collapse_message')
                )

            self.phase_counter += 1
        else:
            # All nodes compromised
            final_text = self.translator.tr('final_state_all_compromised')
            self.update_info(final_text)
            self.log_message(final_text, 'critical')
            QMessageBox.critical(
                self, self.translator.tr('simulation_complete_title'),
                self.translator.tr('final_state_all_compromised')
            )
            self.next_button.setEnabled(False)
            return

    def send_messages(self):
        """
        Simulate sending messages from generals.
        Honest generals send consistent messages.
        Malicious generals send conflicting or false messages.
        """
        self.update_info(self.translator.tr('generals_sending_messages'))
        # Removed logging of normal message sending
        self.message_data = []

        # Each node sends multiple messages per phase
        for _ in range(self.messages_per_step):
            for sender in self.TG.nodes():
                if self.TG.nodes[sender]['status'] in ['detected', 'confused', 'blacklisted']:
                    continue  # Isolated, detected, or blacklisted nodes do not send messages
                receiver = random.choice(list(self.TG.nodes()))
                if sender != receiver and self.TG.nodes[receiver]['status'] not in ['detected', 'confused', 'blacklisted']:
                    if self.TG.nodes[sender]['status'] == 'honest':
                        message = self.translator.tr('attack_at_dawn')
                        color = "green"  # Honest messages in green
                    else:
                        message = random.choice([
                            self.translator.tr('attack_at_dawn'),
                            self.translator.tr('retreat_immediately'),
                            self.translator.tr('send_reinforcements')
                        ])
                        color = "red"  # Malicious messages in red
                    self.message_data.append((sender, receiver, message, color))

        # Start animation
        self.msg_animation_index = 0
        self.progress_bar.setValue(0)  # Reset progress bar
        self.animate_messages()

    def animate_messages(self):
        """Animate messages being sent between nodes."""
        if self.msg_animation_index >= len(self.message_data):
            # Animation complete
            self.canvas.draw()
            self.progress_bar.setValue(100)  # Set progress bar to 100%
            # After animation, perform BFT message verification with adjustable delay
            QTimer.singleShot(self.simulation_speed, self.verify_messages)
            return

        sender, receiver, message, color = self.message_data[self.msg_animation_index]
        start = self.POS[sender]
        end = self.POS[receiver]

        # Calculate adjusted start and end points to stop at node edges
        adjusted_start, adjusted_end = self.calculate_arrow_positions(sender, receiver, start, end)

        arrow = FancyArrowPatch(
            adjusted_start, adjusted_end, arrowstyle='->', mutation_scale=20,
            color=color, alpha=0.8, linewidth=2, zorder=2
        )
        self.ax.add_patch(arrow)
        self.messages.append(arrow)
        self.canvas.draw()

        # Update progress bar
        progress = int((self.msg_animation_index + 1) / len(self.message_data) * 100)
        self.progress_bar.setValue(progress)

        self.msg_animation_index += 1
        # Use simulation_speed for delay between messages
        QTimer.singleShot(self.simulation_speed, self.animate_messages)  # Adjust delay based on slider

    def calculate_arrow_positions(self, sender, receiver, start, end):
        """
        Calculate adjusted start and end positions for arrows to stop at node edges.
        """
        # Define node radius based on image size
        node_radius = 0.05  # Adjust as needed

        # Vector from sender to receiver
        vec = np.array(end) - np.array(start)
        distance = np.linalg.norm(vec)
        if distance == 0:
            return start, end  # Same position, avoid division by zero

        unit_vec = vec / distance

        # Adjust start and end positions
        adjusted_start = np.array(start) + unit_vec * node_radius
        adjusted_end = np.array(end) - unit_vec * node_radius

        return adjusted_start, adjusted_end

    def verify_messages(self):
        """Verify messages to identify malicious nodes."""
        if hasattr(self, 'system_compromised') and self.system_compromised:
            # When the system is compromised, honest nodes cannot detect malicious nodes effectively
            self.update_info(self.translator.tr('system_compromised_info'))
            self.log_message(
                self.translator.tr('system_compromised_log'),
                'critical'
            )
            QMessageBox.critical(
                self, self.translator.tr('system_compromised_title'),
                self.translator.tr('system_compromised_message')
            )
            self.convert_honest_nodes()
            return

        received_messages = {node: [] for node in self.TG.nodes()}
        for sender, receiver, message, color in self.message_data:
            received_messages[receiver].append((sender, message))

        # Identify inconsistencies
        for receiver, msgs in received_messages.items():
            if self.TG.nodes[receiver]['status'] in ['detected', 'confused', 'blacklisted']:
                continue  # Skip detected, confused, or blacklisted nodes

            message_counts = {}
            for sender, msg in msgs:
                message_counts[msg] = message_counts.get(msg, 0) + 1

            if message_counts:
                most_common_message = max(message_counts, key=message_counts.get)
                count = message_counts[most_common_message]
                # Only log critical events, remove info logs for normal operations
                if count < self.detection_threshold * len(msgs):
                    self.log_message(
                        self.translator.tr('inconsistent_messages_detected', receiver=receiver),
                        'warning'
                    )
                    # Identify senders who sent different messages
                    for sender, msg in msgs:
                        if (
                                msg != most_common_message and
                                self.TG.nodes[sender]['status'] not in ['detected', 'confused', 'blacklisted']
                        ):
                            self.detected_malicious.add(sender)
                            self.TG.nodes[sender]['status'] = 'detected'
                            # Overlay a red cross will be handled in plot_graph
                            self.plot_graph()
                            detection_text = self.translator.tr('node_detected', node=sender)
                            self.update_info(detection_text)
                            self.log_message(detection_text, 'critical')
                            QMessageBox.warning(
                                self, self.translator.tr('malicious_node_detected_title'),
                                self.translator.tr('malicious_node_detected_message', node=sender)
                            )

    def convert_honest_nodes(self):
        """Convert honest nodes to a 'confused' state when overwhelmed by malicious majority."""
        for node in self.TG.nodes():
            if self.TG.nodes[node]['status'] == 'honest':
                self.TG.nodes[node]['status'] = 'confused'
                self.TG.nodes[node]['image'] = self.confused_img
                self.TG.nodes[node]['size'] = 0.03  # Ensure size consistency
        self.plot_graph()
        self.update_info(self.translator.tr('honest_overwhelmed_info'))
        self.log_message(
            self.translator.tr('honest_overwhelmed_log'),
            'critical'
        )
        QMessageBox.information(
            self, self.translator.tr('honest_overwhelmed_title'),
            self.translator.tr('honest_overwhelmed_message')
        )

    def plot_graph(self, initial=False):
        """Plot the current state of the graph."""
        self.ax.clear()
        self.ax.set_title(self.translator.tr('graph_title'), fontsize=18)
        self.ax.axis('off')

        # Draw edges
        nx.draw_networkx_edges(
            self.TG, pos=self.POS, edge_color="lightgray", ax=self.ax
        )

        # Draw nodes with images
        for n in self.TG.nodes():
            x, y = self.POS[n]
            image = self.TG.nodes[n]['image']
            size = self.TG.nodes[n]['size']
            ab = AnnotationBbox(
                OffsetImage(image, zoom=size), (x, y), frameon=False, zorder=1
            )
            self.ax.add_artist(ab)
            # Add node labels
            self.ax.text(x, y - 0.05, str(n), horizontalalignment='center', fontsize=8, zorder=4)

        # Draw existing messages
        for arrow in self.messages:
            self.ax.add_patch(arrow)

        # Draw detected malicious nodes with a red cross
        for n in self.detected_malicious:
            x, y = self.POS[n]
            # Draw a red cross
            cross_size = 0.05
            line1 = Line2D([x - cross_size, x + cross_size], [y - cross_size, y + cross_size],
                           color='red', linewidth=2, zorder=5)
            line2 = Line2D([x - cross_size, x + cross_size], [y + cross_size, y - cross_size],
                           color='red', linewidth=2, zorder=5)
            self.ax.add_line(line1)
            self.ax.add_line(line2)

        # Draw blacklisted nodes with blacklisted.png and grey circle overlay
        for n in self.TG.nodes():
            if self.TG.nodes[n]['status'] == 'blacklisted':
                x, y = self.POS[n]
                # Overlay a grey circle to signify blacklisting
                circ = plt.Circle((x, y), 0.07, color='grey', fill=True, alpha=0.5, zorder=4)
                self.ax.add_patch(circ)

        self.canvas.draw()


def main():
    app = QApplication(sys.argv)

    # Initialize Translator
    translator = Translator()

    # If language is not set in config, prompt the user
    if not os.path.exists(translator.config_file) or not translator.translations:
        translator.select_language()
        translator.load_language()

    # Initialize and show the emulator
    # Prompt user for the number of nodes
    n_nodes, ok = QInputDialog.getInt(
        None, "Input", translator.tr('input_nodes_prompt'),
        value=12, min=3
    )
    if not ok:
        n_nodes = 12  # Default value if user cancels

    emulator = ByzantiumEmulator(translator, n_nodes=n_nodes)
    emulator.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
