#!/usr/bin/env python3
"""
Create an interactive HTML dashboard to visualize all evaluation plots.

Generates a user-friendly interface with tabs for different views and
organized navigation through all 72 plots.
"""

import json
from pathlib import Path
from datetime import datetime


# Configuration
PLOTS_BASE = Path("02_OUTPUTS/TIMIT_Outputs/plots/01_Evaluation_plots")
OUTPUT_FILE = Path("02_OUTPUTS/TIMIT_Outputs/plots/evaluation_dashboard.html")
SUMMARY_FILES = ["02_OUTPUTS/TIMIT_Outputs/probes/linear/linear_evaluation_summary.json",
                 "02_OUTPUTS/TIMIT_Outputs/probes/mlp_1x200/mlp_1x200_evaluation_summary.json"]

MODELS = ["HUBERT_BASE", "HUBERT_LARGE", "WAV2VEC2_BASE", "WAV2VEC2_LARGE", "WAVLM_BASE", "WAVLM_LARGE"]
FEATURES = ["voiced", "fricative", "nasal"]
ARCHITECTURES = ["linear", "mlp_1x200"]
METRICS = ["01_Accuracy", "02_F1_Score"]


def get_statistics():
    """Extract summary statistics from evaluation JSON files."""
    stats = {}
    
    for summary_path in SUMMARY_FILES:
        path = Path(summary_path)
        if not path.exists():
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
        
        arch = path.stem.replace("_evaluation_summary", "")
        stats[arch] = {
            "total_probes": len(data),
            "models": len(set(r["model"] for r in data)),
            "layers": {},
            "avg_accuracy": sum(r["metrics"]["accuracy"] for r in data) / len(data),
            "avg_f1": sum(r["metrics"]["f1"] for r in data) / len(data),
        }
        
        # Count layers per model
        for model in MODELS:
            model_data = [r for r in data if r["model"] == model]
            if model_data:
                stats[arch]["layers"][model] = max(r["layer"] for r in model_data) + 1
    
    return stats


def generate_html():
    """Generate the complete HTML dashboard."""
    
    stats = get_statistics()
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phonetic Feature Probe Evaluation Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.95;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        
        .stat-card h3 {{
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        
        .stat-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        
        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            overflow-x: auto;
        }}
        
        .tab {{
            padding: 15px 30px;
            cursor: pointer;
            background: #f8f9fa;
            border: none;
            font-size: 1em;
            font-weight: 500;
            color: #666;
            transition: all 0.3s;
            white-space: nowrap;
        }}
        
        .tab:hover {{
            background: #e9ecef;
            color: #667eea;
        }}
        
        .tab.active {{
            background: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }}
        
        .tab-content {{
            display: none;
            padding: 30px;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .filter-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        
        .filter-group {{
            flex: 1;
            min-width: 200px;
        }}
        
        .filter-group label {{
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #495057;
        }}
        
        .filter-group select {{
            width: 100%;
            padding: 10px;
            border: 2px solid #dee2e6;
            border-radius: 5px;
            font-size: 1em;
            background: white;
            cursor: pointer;
            transition: border-color 0.3s;
        }}
        
        .filter-group select:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
        }}
        
        .plot-card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .plot-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        }}
        
        .plot-card img {{
            width: 100%;
            height: auto;
            display: block;
            cursor: pointer;
        }}
        
        .plot-card .plot-info {{
            padding: 15px;
            background: #f8f9fa;
        }}
        
        .plot-card .plot-title {{
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .plot-card .plot-meta {{
            font-size: 0.9em;
            color: #6c757d;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-right: 5px;
        }}
        
        .badge-model {{ background: #e7f5ff; color: #1864ab; }}
        .badge-feature {{ background: #fff3cd; color: #856404; }}
        .badge-arch {{ background: #d4edda; color: #155724; }}
        
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        
        .modal.active {{
            display: flex;
        }}
        
        .modal img {{
            max-width: 95%;
            max-height: 95%;
            border-radius: 10px;
        }}
        
        .modal-close {{
            position: absolute;
            top: 20px;
            right: 30px;
            color: white;
            font-size: 40px;
            cursor: pointer;
            transition: color 0.3s;
        }}
        
        .modal-close:hover {{
            color: #667eea;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .plot-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Phonetic Feature Probe Evaluation Dashboard</h1>
            <p>Interactive visualization of probe performance across models, layers, and features</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Plots</h3>
                <div class="value">72</div>
            </div>
            <div class="stat-card">
                <h3>Models Evaluated</h3>
                <div class="value">6</div>
            </div>
            <div class="stat-card">
                <h3>Probe Architectures</h3>
                <div class="value">2</div>
            </div>
            <div class="stat-card">
                <h3>Features Probed</h3>
                <div class="value">3</div>
            </div>
"""

    # Add architecture-specific statistics
    for arch, arch_stats in stats.items():
        html += f"""
            <div class="stat-card">
                <h3>{arch.upper()} - Avg Accuracy</h3>
                <div class="value">{arch_stats['avg_accuracy']:.3f}</div>
            </div>
            <div class="stat-card">
                <h3>{arch.upper()} - Avg F1 Score</h3>
                <div class="value">{arch_stats['avg_f1']:.3f}</div>
            </div>
"""

    html += """
        </div>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('variant1')">📊 Features per Model</button>
            <button class="tab" onclick="switchTab('variant2')">📈 Models per Feature</button>
            <button class="tab" onclick="switchTab('variant3')">🔄 Architecture Comparison</button>
            <button class="tab" onclick="switchTab('overview')">🗂️ Complete Overview</button>
        </div>
"""

    # Generate tab contents
    html += generate_variant1_tab()
    html += generate_variant2_tab()
    html += generate_variant3_tab()
    html += generate_overview_tab()

    html += f"""
        <div class="footer">
            Generated on {generation_time} | Total Probes: {sum(s['total_probes'] for s in stats.values())} | 
            Models: {', '.join(MODELS)}
        </div>
    </div>
    
    <div class="modal" id="imageModal" onclick="closeModal()">
        <span class="modal-close">&times;</span>
        <img id="modalImage" src="" alt="Full size plot">
    </div>
    
    <script>
        function switchTab(tabName) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }}
        
        function openModal(imgSrc) {{
            document.getElementById('imageModal').classList.add('active');
            document.getElementById('modalImage').src = imgSrc;
        }}
        
        function closeModal() {{
            document.getElementById('imageModal').classList.remove('active');
        }}
        
        function filterPlots(metric, architecture, model, feature) {{
            const plots = document.querySelectorAll('.plot-card');
            
            plots.forEach(plot => {{
                const plotMetric = plot.dataset.metric;
                const plotArch = plot.dataset.architecture;
                const plotModel = plot.dataset.model;
                const plotFeature = plot.dataset.feature;
                
                let show = true;
                
                if (metric !== 'all' && plotMetric !== metric) show = false;
                if (architecture !== 'all' && plotArch !== architecture) show = false;
                if (model !== 'all' && plotModel !== model) show = false;
                if (feature !== 'all' && plotFeature !== feature) show = false;
                
                plot.style.display = show ? 'block' : 'none';
            }});
        }}
    </script>
</body>
</html>
"""
    
    return html


def generate_variant1_tab():
    """Generate Variant 1: Features per Model per Architecture."""
    html = '<div id="variant1" class="tab-content active">'
    html += '<h2 style="margin-bottom: 20px;">📊 All Features per Model per Architecture</h2>'
    html += '<p style="margin-bottom: 20px; color: #6c757d;">Compare how different phonetic features (voiced, fricative, nasal) perform across layers for each model and architecture.</p>'
    
    for metric in METRICS:
        metric_name = "Accuracy" if metric == "01_Accuracy" else "F1 Score"
        html += f'<h3 style="margin: 30px 0 20px 0; color: #667eea;">📈 {metric_name}</h3>'
        html += '<div class="plot-grid">'
        
        for arch in ARCHITECTURES:
            for model in MODELS:
                plot_path = PLOTS_BASE / metric / arch / f"{model}.png"
                if plot_path.exists():
                    rel_path = str(plot_path)
                    html += f"""
                    <div class="plot-card">
                        <img src="{rel_path}" alt="{model} - {arch}" onclick="openModal('{rel_path}')">
                        <div class="plot-info">
                            <div class="plot-title">{model}</div>
                            <div class="plot-meta">
                                <span class="badge badge-arch">{arch}</span>
                                <span class="badge badge-feature">All Features</span>
                            </div>
                        </div>
                    </div>
                    """
        
        html += '</div>'
    
    html += '</div>'
    return html


def generate_variant2_tab():
    """Generate Variant 2: Models per Feature per Architecture."""
    html = '<div id="variant2" class="tab-content">'
    html += '<h2 style="margin-bottom: 20px;">📈 All Models per Feature per Architecture</h2>'
    html += '<p style="margin-bottom: 20px; color: #6c757d;">Compare how different models perform for each phonetic feature across layers.</p>'
    
    for metric in METRICS:
        metric_name = "Accuracy" if metric == "01_Accuracy" else "F1 Score"
        html += f'<h3 style="margin: 30px 0 20px 0; color: #667eea;">📊 {metric_name}</h3>'
        html += '<div class="plot-grid">'
        
        for arch in ARCHITECTURES:
            for feature in FEATURES:
                plot_path = PLOTS_BASE / metric / arch / f"{feature}.png"
                if plot_path.exists():
                    rel_path = str(plot_path)
                    html += f"""
                    <div class="plot-card">
                        <img src="{rel_path}" alt="{feature} - {arch}" onclick="openModal('{rel_path}')">
                        <div class="plot-info">
                            <div class="plot-title">{feature.capitalize()}</div>
                            <div class="plot-meta">
                                <span class="badge badge-arch">{arch}</span>
                                <span class="badge badge-model">All Models</span>
                            </div>
                        </div>
                    </div>
                    """
        
        html += '</div>'
    
    html += '</div>'
    return html


def generate_variant3_tab():
    """Generate Variant 3: Architectures per Model per Feature."""
    html = '<div id="variant3" class="tab-content">'
    html += '<h2 style="margin-bottom: 20px;">🔄 Architecture Comparison per Model per Feature</h2>'
    html += '<p style="margin-bottom: 20px; color: #6c757d;">Compare linear vs MLP probe architectures for each model-feature combination.</p>'
    
    for metric in METRICS:
        metric_name = "Accuracy" if metric == "01_Accuracy" else "F1 Score"
        html += f'<h3 style="margin: 30px 0 20px 0; color: #667eea;">🎯 {metric_name}</h3>'
        html += '<div class="plot-grid">'
        
        for model in MODELS:
            for feature in FEATURES:
                plot_path = PLOTS_BASE / metric / model / f"{feature}.png"
                if plot_path.exists():
                    rel_path = str(plot_path)
                    html += f"""
                    <div class="plot-card">
                        <img src="{rel_path}" alt="{model} - {feature}" onclick="openModal('{rel_path}')">
                        <div class="plot-info">
                            <div class="plot-title">{model} - {feature.capitalize()}</div>
                            <div class="plot-meta">
                                <span class="badge badge-model">{model}</span>
                                <span class="badge badge-feature">{feature}</span>
                                <span class="badge badge-arch">Both Architectures</span>
                            </div>
                        </div>
                    </div>
                    """
        
        html += '</div>'
    
    html += '</div>'
    return html


def generate_overview_tab():
    """Generate overview tab with all plots and filters."""
    html = '<div id="overview" class="tab-content">'
    html += '<h2 style="margin-bottom: 20px;">🗂️ Complete Overview - All 72 Plots</h2>'
    
    # Add filter bar
    html += """
    <div class="filter-bar">
        <div class="filter-group">
            <label>Metric:</label>
            <select onchange="filterPlots(this.value, 
                document.getElementById('archFilter').value,
                document.getElementById('modelFilter').value,
                document.getElementById('featureFilter').value)">
                <option value="all">All Metrics</option>
                <option value="01_Accuracy">Accuracy</option>
                <option value="02_F1_Score">F1 Score</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Architecture:</label>
            <select id="archFilter" onchange="filterPlots(
                document.querySelector('.filter-bar select').value,
                this.value,
                document.getElementById('modelFilter').value,
                document.getElementById('featureFilter').value)">
                <option value="all">All Architectures</option>
                <option value="linear">Linear</option>
                <option value="mlp_1x200">MLP 1x200</option>
            </select>
        </div>
        <div class="filter-group">
            <label>Model:</label>
            <select id="modelFilter" onchange="filterPlots(
                document.querySelector('.filter-bar select').value,
                document.getElementById('archFilter').value,
                this.value,
                document.getElementById('featureFilter').value)">
                <option value="all">All Models</option>
"""
    
    for model in MODELS:
        html += f'                <option value="{model}">{model}</option>\n'
    
    html += """
            </select>
        </div>
        <div class="filter-group">
            <label>Feature:</label>
            <select id="featureFilter" onchange="filterPlots(
                document.querySelector('.filter-bar select').value,
                document.getElementById('archFilter').value,
                document.getElementById('modelFilter').value,
                this.value)">
                <option value="all">All Features</option>
                <option value="voiced">Voiced</option>
                <option value="fricative">Fricative</option>
                <option value="nasal">Nasal</option>
            </select>
        </div>
    </div>
    """
    
    html += '<div class="plot-grid">'
    
    # Add all plots with metadata
    for metric in METRICS:
        for arch in ARCHITECTURES:
            # Model plots
            for model in MODELS:
                plot_path = PLOTS_BASE / metric / arch / f"{model}.png"
                if plot_path.exists():
                    rel_path = str(plot_path)
                    html += f"""
                    <div class="plot-card" data-metric="{metric}" data-architecture="{arch}" 
                         data-model="{model}" data-feature="all">
                        <img src="{rel_path}" alt="{model} - {arch}" onclick="openModal('{rel_path}')">
                        <div class="plot-info">
                            <div class="plot-title">{model}</div>
                            <div class="plot-meta">
                                <span class="badge badge-arch">{arch}</span>
                                <span class="badge badge-feature">All Features</span>
                            </div>
                        </div>
                    </div>
                    """
            
            # Feature plots
            for feature in FEATURES:
                plot_path = PLOTS_BASE / metric / arch / f"{feature}.png"
                if plot_path.exists():
                    rel_path = str(plot_path)
                    html += f"""
                    <div class="plot-card" data-metric="{metric}" data-architecture="{arch}" 
                         data-model="all" data-feature="{feature}">
                        <img src="{rel_path}" alt="{feature} - {arch}" onclick="openModal('{rel_path}')">
                        <div class="plot-info">
                            <div class="plot-title">{feature.capitalize()}</div>
                            <div class="plot-meta">
                                <span class="badge badge-arch">{arch}</span>
                                <span class="badge badge-model">All Models</span>
                            </div>
                        </div>
                    </div>
                    """
        
        # Architecture comparison plots
        for model in MODELS:
            for feature in FEATURES:
                plot_path = PLOTS_BASE / metric / model / f"{feature}.png"
                if plot_path.exists():
                    rel_path = str(plot_path)
                    html += f"""
                    <div class="plot-card" data-metric="{metric}" data-architecture="both" 
                         data-model="{model}" data-feature="{feature}">
                        <img src="{rel_path}" alt="{model} - {feature}" onclick="openModal('{rel_path}')">
                        <div class="plot-info">
                            <div class="plot-title">{model} - {feature.capitalize()}</div>
                            <div class="plot-meta">
                                <span class="badge badge-model">{model}</span>
                                <span class="badge badge-feature">{feature}</span>
                            </div>
                        </div>
                    </div>
                    """
    
    html += '</div></div>'
    return html


def main():
    """Generate the dashboard."""
    print("=" * 70)
    print("CREATING INTERACTIVE EVALUATION DASHBOARD")
    print("=" * 70)
    
    print("\n[1/2] Gathering statistics and generating HTML...")
    html_content = generate_html()
    
    print("[2/2] Writing dashboard file...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("\n" + "=" * 70)
    print("DASHBOARD CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nDashboard location: {OUTPUT_FILE}")
    print(f"Open in browser: file://{OUTPUT_FILE.absolute()}")
    print("\nFeatures:")
    print("  ✓ Interactive tabs for different view types")
    print("  ✓ Dynamic filtering in overview tab")
    print("  ✓ Click images for full-size view")
    print("  ✓ Performance statistics summary")
    print("  ✓ Responsive design for all screen sizes")


if __name__ == "__main__":
    main()
