# generate_report.py
"""
Attrahere CLI Report Generator - Phase 3 Implementation

Generates professional ML code analysis reports following the specification
defined in specs/cli_report_spec.md

This script transforms detection results into formatted CLI output with:
- ASCII art header
- Executive summary tables
- Categorized findings by impact level
- Economic impact analysis
- Actionable recommendations
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analysis_core.ml_analyzer.detectors.data_flow_contamination_detector import (
    DataFlowContaminationDetector,
)
from analysis_core.ml_analyzer.detectors.temporal_leakage_detector import (
    TemporalLeakageDetector,
)
from analysis_core.ml_analyzer.detectors.magic_numbers_detector import (
    MagicNumberExtractor,
)
from analysis_core.ml_analyzer.detectors.preprocessing_leakage_detector import (
    PreprocessingLeakageDetector,
)
from analysis_core.ml_analyzer.ast_engine import MLSemanticAnalyzer

# 1. Dati DUMMY per lo sviluppo della UI (verranno sostituiti con dati reali)
DUMMY_RESULTS = {
    "files_analyzed": 12,
    "patterns_identified": 27,
    "high_impact": 8,
    "medium_impact": 12,
    "low_impact": 7,
    "avg_confidence": 82.5,
    "analysis_time_sec": 2.3,
    "target_project": "yolov5",
    "project_type": "Computer Vision Pipeline",
}

DUMMY_PATTERN_DISTRIBUTION = {
    "Data Flow Contamination": {"count": 6, "impact": "🚨"},
    "GPU Memory Leak": {"count": 4, "impact": "🚨"},
    "Test Set Contamination": {"count": 5, "impact": "⚠️"},
    "Temporal Leakage": {"count": 3, "impact": "🚨"},
    "Inefficient Data Loading": {"count": 4, "impact": "⚠️"},
    "Hardcoded Thresholds": {"count": 3, "impact": "💡"},
    "Feature Engineering Leakage": {"count": 2, "impact": "⚠️"},
}

# =====================================================================
# REAL DATA INTEGRATION - Task 2 Implementation
# =====================================================================


def execute_with_monitoring(target_file):
    """
    Esegue uno script come subprocess e cattura l'output JSON del monitor.
    """
    console = Console()
    console.print(f"⚡ Esecuzione monitorata di: {target_file}", style="dim")

    try:
        # Esegui lo script usando lo stesso interprete Python
        result = subprocess.run(
            [sys.executable, target_file],
            capture_output=True,
            text=True,
            check=False,  # Non sollevare eccezioni per errori nello script
        )

        # Cerca la riga JSON del nostro monitor nell'output
        for line in result.stdout.splitlines():
            # Cerca la chiave unica del nostro monitor
            if '"monitor": "attrahere"' in line:
                try:
                    monitor_data = json.loads(line)
                    console.print("✅ Dati di monitoring catturati.", style="green")
                    return monitor_data
                except json.JSONDecodeError:
                    console.print(
                        "⚠️ Attenzione: Trovata riga del monitor, ma parsing JSON fallito.",
                        style="yellow",
                    )
                    return {}

        console.print(
            "⚠️ Attenzione: Nessun output del monitor Attrahere trovato.", style="yellow"
        )
        return {}  # Ritorna un dizionario vuoto se non trova nulla

    except Exception as e:
        console.print(
            f"❌ Errore durante l'esecuzione monitorata: {e}", style="bold red"
        )
        return {}


def run_full_analysis(target_file, context):
    """
    Orchestrate complete analysis: static detection + runtime monitoring

    Args:
        target_file (str): Path to Python file to analyze

    Returns:
        tuple: (patterns_list, monitor_logs_dict)
    """
    console = Console()
    console.print(f"🔍 Analyzing: {target_file}", style="bold blue")

    try:
        # Initialize ML analyzer and detectors
        semantic_analyzer = MLSemanticAnalyzer()
        dataflow_detector = DataFlowContaminationDetector()
        temporal_detector = TemporalLeakageDetector(analysis_context=context)
        magic_detector = MagicNumberExtractor()
        preprocessing_detector = PreprocessingLeakageDetector()

        # Run static analysis
        console.print("📊 Running static analysis...", style="dim")
        analysis_result = semantic_analyzer.analyze_file(Path(target_file))

        # Run all detectors
        console.print("🔍 DataFlow contamination detection...", style="dim")
        dataflow_patterns = dataflow_detector.detect_patterns(analysis_result)

        console.print("⏱️ Temporal leakage detection...", style="dim")
        temporal_patterns = temporal_detector.detect_patterns(analysis_result)
        
        console.print("🔢 Magic number detection...", style="dim")
        magic_patterns = magic_detector.detect_patterns(analysis_result)
        
        console.print("⚗️ Preprocessing leakage detection...", style="dim")
        preprocessing_patterns = preprocessing_detector.detect_patterns(analysis_result)

        # Combine patterns from all detectors
        patterns = dataflow_patterns + temporal_patterns + magic_patterns + preprocessing_patterns

        console.print(
            f"✅ Static analysis complete: {len(patterns)} patterns detected",
            style="green",
        )

        # Runtime monitoring integration
        console.print("⚡ Running target with monitoring...", style="dim")
        monitor_logs = execute_with_monitoring(target_file)

        return patterns, monitor_logs

    except Exception as e:
        console.print(f"❌ Analysis failed: {e}", style="bold red")
        return [], {}


def transform_results(patterns, monitor_logs, target_file):
    """
    Transform raw analysis results into report-ready data structures

    Args:
        patterns (list): ML patterns from detector
        monitor_logs (dict): Runtime monitoring data
        target_file (str): Original target file path

    Returns:
        tuple: (summary_data, pattern_distribution, findings_list)
    """
    # Calculate summary metrics
    total_patterns = len(patterns)

    # Classify patterns by severity
    high_impact = sum(1 for p in patterns if p.severity.name == "HIGH")
    medium_impact = sum(1 for p in patterns if p.severity.name == "MEDIUM")
    low_impact = sum(1 for p in patterns if p.severity.name == "LOW")

    # Calculate average confidence
    avg_confidence = (
        sum(p.confidence for p in patterns) / total_patterns
        if total_patterns > 0
        else 0.0
    )

    # Extract project info from file path
    project_name = Path(target_file).stem

    summary_data = {
        "files_analyzed": 1,  # Single file for now
        "patterns_identified": total_patterns,
        "high_impact": high_impact,
        "medium_impact": medium_impact,
        "low_impact": low_impact,
        "avg_confidence": round(avg_confidence * 100, 1),  # Convert to percentage
        "analysis_time_sec": monitor_logs.get("duration_seconds", 0.0),
        "target_project": project_name,
        "project_type": "ML Code Analysis",
    }

    # Build pattern distribution by type with better categorization
    pattern_distribution = {}
    for pattern in patterns:
        # Categorize pattern types more clearly
        if "pipeline_contamination" in pattern.pattern_type:
            category = "Data Flow Contamination"
        elif "temporal" in pattern.pattern_type:
            category = "Temporal Leakage"
        elif "feature_engineering" in pattern.pattern_type:
            category = "Feature Engineering Leakage"
        elif "cv_preprocessing" in pattern.pattern_type:
            category = "Cross-Validation Issues"
        elif "global_statistics" in pattern.pattern_type:
            category = "Global Statistics Leakage"
        else:
            category = pattern.pattern_type.replace("_", " ").title()

        if category not in pattern_distribution:
            # Map severity to emoji
            severity_emoji = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "💡"}.get(
                pattern.severity.name, "🔍"
            )

            pattern_distribution[category] = {"count": 0, "impact": severity_emoji}

        pattern_distribution[category]["count"] += 1

    # Create findings list with key details
    findings_list = []
    for pattern in patterns:
        finding = {
            "type": pattern.pattern_type,
            "severity": pattern.severity.name,
            "file": target_file,
            "line": pattern.line_number,
            "message": pattern.message,
            "confidence": pattern.confidence,
        }
        findings_list.append(finding)

    return summary_data, pattern_distribution, findings_list


def analyze_and_report(target_file, context="time-series"):
    """
    Complete pipeline: analyze file and generate report with real data

    Args:
        target_file (str): Path to Python file to analyze
    """
    console = Console()

    # Step 1: Run analysis
    patterns, monitor_logs = run_full_analysis(target_file, context)

    if not patterns:
        console.print("ℹ️  No patterns detected or analysis failed", style="yellow")
        return

    # Step 2: Transform results
    summary_data, pattern_distribution, findings_list = transform_results(
        patterns, monitor_logs, target_file
    )

    # Step 3: Generate report with real data
    console.print("\n🎯 Generating Report with Real Data", style="bold green")
    console.print("=" * 50)

    print_report_header(summary_data)
    print_summary_table(summary_data)

    # Print real findings
    console.print("🚨 REAL FINDINGS DETECTED", style="bold red")
    console.print("─────────────────────────")

    for finding in findings_list:
        severity_emoji = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "💡"}.get(
            finding["severity"], "🔍"
        )
        console.print(
            f"• {severity_emoji} {finding['type']} - {Path(finding['file']).name}:{finding['line']}"
        )
        console.print(f"  💬 {finding['message']}")
        console.print(f"  🎯 Confidence: {finding['confidence']:.0%}")
        console.print()

    # Recommendations based on real findings
    if findings_list:
        console.print("📋 RACCOMANDAZIONI REAL-TIME", style="bold yellow")
        console.print("──────────────────────────")

        high_priority = [f for f in findings_list if f["severity"] == "HIGH"]
        medium_priority = [f for f in findings_list if f["severity"] == "MEDIUM"]

        if high_priority:
            console.print(f"1. PRIORITÀ ALTA: Fix {len(high_priority)} critical issues")
        if medium_priority:
            console.print(f"2. PRIORITÀ MEDIA: Address {len(medium_priority)} warnings")

        console.print(f"3. TOTAL: {len(findings_list)} patterns require attention")
        console.print()


def generate_cli_header(results):
    """Generate clean, professional CLI header"""
    header = """================================
           ATTRahere           
      Semantic Code Analysis   
================================
📅 Report: {timestamp}
🎯 Target: {project_name}
🔍 Analysis: {analysis_type}
📊 Version: Sprint 4 MVP
================================""".format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        project_name=results["target_project"],
        analysis_type="Data Flow Patterns",
    )
    return header


def print_report_header(results):
    """Print clean, readable header with project info"""
    console = Console()

    # Clean header
    header = generate_cli_header(results)
    console.print(header, style="bold blue")
    console.print()


def print_summary_table(results):
    """Print executive summary table with key metrics"""
    console = Console()

    console.print("📊 RIEPILOGO ESECUTIVO", style="bold yellow")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("METRICA", style="dim", width=35)
    table.add_column("VALORE", justify="right", width=13)
    table.add_column("STATUS", justify="center", width=14)

    table.add_row("File analizzati", str(results["files_analyzed"]), "✅")
    table.add_row("Pattern identificati", str(results["patterns_identified"]), "🔍")
    table.add_row("Pattern ad alto impatto", str(results["high_impact"]), "🚨")
    table.add_row("Pattern a medio impatto", str(results["medium_impact"]), "⚠️")
    table.add_row("Pattern a basso impatto", str(results["low_impact"]), "💡")
    table.add_row("Confidence media", f"{results['avg_confidence']}%", "🎯")
    table.add_row("Tempo di analisi", f"{results['analysis_time_sec']} sec", "⚡")

    console.print(table)
    console.print()


def print_pattern_distribution_table(pattern_data):
    """Print pattern distribution by type and impact"""
    console = Console()

    console.print("🎯 DISTRIBUZIONE PATTERN PER TIPO", style="bold yellow")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("TIPO PATTERN", style="dim", width=36)
    table.add_column("COUNT", justify="center", width=7)
    table.add_column("IMPATTO", justify="center", width=12)

    for pattern_type, data in pattern_data.items():
        table.add_row(pattern_type, str(data["count"]), data["impact"])

    console.print(table)
    console.print()


def print_critical_findings_summary():
    """Print concise critical findings summary"""
    console = Console()

    console.print("🚨 FINDINGS CRITICI (8)", style="bold red")
    console.print("────────────────────────")
    console.print("• Data Flow Contamination - train.py:142")
    console.print("• GPU Memory Leak - train.py:89")
    console.print("• Temporal Leakage - datasets.py:178")
    console.print()


def print_file_analysis_table():
    """Print file-by-file analysis summary"""
    console = Console()

    console.print("📈 ANALISI PER FILE", style="bold yellow")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("FILE", style="dim", width=30)
    table.add_column("PATTERNS", justify="center", width=10)
    table.add_column("🚨", justify="center", width=6)
    table.add_column("⚠️", justify="center", width=6)
    table.add_column("💡", justify="center", width=6)

    # Dummy file data
    files_data = [
        ("train.py", 8, 3, 4, 1),
        ("utils/datasets.py", 7, 2, 3, 2),
        ("utils/metrics.py", 5, 1, 2, 2),
        ("utils/loss.py", 4, 1, 2, 1),
        ("models/common.py", 3, 1, 1, 1),
    ]

    for file_name, total, high, medium, low in files_data:
        table.add_row(file_name, str(total), str(high), str(medium), str(low))

    console.print(table)
    console.print()


def print_economic_impact():
    """Print economic impact analysis"""
    console = Console()

    console.print("💰 STIMA IMPATTO ECONOMICO", style="bold yellow")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("CATEGORIA", style="dim", width=32)
    table.add_column("COSTO ATTUALE", justify="right", width=17)
    table.add_column("POTENZIALE\nRISPARMIO", justify="right", width=17)

    table.add_row(
        "🚨 Data Leakage Issues\n   (GPU waste + inaccurate models)",
        "$1,200/mese",
        "$900/mese",
    )
    table.add_row(
        "⚠️ Performance Inefficiencies\n   (Longer training times)",
        "$800/mese",
        "$500/mese",
    )
    table.add_row(
        "💡 Technical Debt\n   (Maintenance overhead)", "$400/mese", "$200/mese"
    )
    table.add_row("📊 TOTALE", "$2,400/mese", "$1,600/mese", style="bold")

    console.print(table)
    console.print()
    console.print(
        "💰 ROI POTENZIALE: 67% riduzione costi infrastrutturali", style="bold green"
    )
    console.print()


def print_recommendations():
    """Print concise prioritized recommendations"""
    console = Console()

    console.print("📋 RACCOMANDAZIONI", style="bold yellow")
    console.print("───────────────────")
    console.print("1. PRIORITÀ ALTA: Fix data leakage in train.py")
    console.print("2. PRIORITÀ MEDIA: Optimize data loading")
    console.print("3. PRIORITÀ BASSA: Replace magic numbers")
    console.print()


def print_footer():
    """Print report footer with next steps"""
    console = Console()

    console.print("🏁 FOOTER E PROSSIMI STEP", style="bold yellow")
    console.print()

    footer_content = """🎉 ANALISI COMPLETATA CON SUCCESSO!

Prossimi passi raccomandati:
1. Implementare i fix per i pattern ad alto impatto
2. Rianalizzare il codice dopo le correzioni  
3. Monitorare le metriche di performance
4. Programmare analisi periodiche

📞 Supporto: docs.attrahere.io/sprint4-mvp
🐛 Bug Report: github.com/attrahere/platform/issues

"Better code today, better models tomorrow." 🚀"""

    panel = Panel(footer_content, border_style="green", padding=(1, 2))
    console.print(panel)


def generate_simplified_report():
    """Generate simplified, clean CLI report"""
    console = Console()

    # Clean header
    print_report_header(DUMMY_RESULTS)

    # Executive summary
    print_summary_table(DUMMY_RESULTS)

    # Critical findings summary
    print_critical_findings_summary()

    # Recommendations
    print_recommendations()


def generate_full_report():
    """Generate complete CLI report with all sections"""
    console = Console()

    # Header
    print_report_header(DUMMY_RESULTS)

    # Executive Summary
    print_summary_table(DUMMY_RESULTS)

    # Pattern Distribution
    print_pattern_distribution_table(DUMMY_PATTERN_DISTRIBUTION)

    # Critical Finding Sample
    print_critical_findings_summary()

    # File Analysis
    print_file_analysis_table()

    # Economic Impact
    print_economic_impact()

    # Recommendations
    print_recommendations()

    # Footer
    print_footer()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Attrahere ML Code Analysis Report Generator"
    )
    parser.add_argument(
        "--mode",
        choices=["dummy", "real"],
        default="real",
        help="Use dummy data or real analysis (default: real)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="tests/validation_suite/scaler_incorrect.py",
        help="Target file for real analysis",
    )
    parser.add_argument(
        "--context", 
        choices=["time-series", "computer-vision", "nlp"],
        default="time-series",
        help="ML domain context for analysis (default: time-series)"
    )

    args = parser.parse_args()

    if args.mode == "dummy":
        # Use simplified report with dummy data (for UI development)
        generate_simplified_report()
    else:
        # Run real analysis on target file
        analyze_and_report(args.target, args.context)
