"""Export and file I/O utilities."""

import json
import csv
from datetime import datetime


class DataExporter:
    """Handle export of training data and metrics."""
    
    @staticmethod
    def export_metrics_json(metrics_data, filename=None):
        """Export metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(metrics_data, f, indent=2)
        return filename

    @staticmethod
    def export_training_csv(history_data, filename=None):
        """Export training history to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_history_{timestamp}.csv"
        
        if not history_data or not history_data[0]:
            return filename
        
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(history_data[0].keys())
            for row in history_data:
                writer.writerow(row.values())
        return filename

    @staticmethod
    def export_model_state(state, filename=None):
        """Export model parameters and state."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_state_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(state, f, indent=2)
        return filename
