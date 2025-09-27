#!/usr/bin/env python3
"""
🎯 Validation Suite Runner - Fase 1: Solidità Inattaccabile

Questo script valida automaticamente l'accuratezza del sistema di analisi:
- Clean code: deve restituire 0 errori
- Problematic code: deve restituire esattamente 1 errore del tipo atteso
- Edge cases: test di robustezza e intelligenza del sistema

Target: >99% accuratezza, <1% falsi positivi
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

# Add analysis_core to path (corrected for V3 unified structure)
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_core.ml_analyzer.analyzer import MLCodeAnalyzer


@dataclass
class ValidationResult:
    file_path: str
    expected_errors: int
    expected_types: List[str]
    actual_errors: int
    actual_types: List[str]
    success: bool
    analysis_time_ms: float
    details: Dict[str, Any]


class ValidationSuite:
    """Test suite per validare accuratezza del motore di analisi"""

    def __init__(self):
        self.analyzer = MLCodeAnalyzer()
        self.results: List[ValidationResult] = []

        # Test cases configuration
        self.test_cases = {
            "clean_code": {
                "clean_preprocessing.py": {"errors": 0, "types": []},
                "clean_reproducibility.py": {"errors": 0, "types": []},
                "clean_gpu_usage.py": {"errors": 0, "types": []},
            },
            "problematic_code": {
                "data_leakage_preprocessing.py": {
                    "errors": 1,
                    "types": ["data_leakage_preprocessing"],
                },
                "missing_random_state.py": {
                    "errors": 1,
                    "types": ["missing_random_seed"],
                },
                "gpu_memory_leak.py": {"errors": 1, "types": ["gpu_memory_leak"]},
            },
            "edge_cases": {
                "global_seed_set.py": {"errors": 0, "types": []},
                "contextual_fixes.py": {"errors": 0, "types": []},
            },
        }

    def run_full_validation(self) -> Dict[str, Any]:
        """Esegue la validazione completa e calcola le metriche"""
        print("🎯 Starting Validation Suite - Fase 1: Solidità Inattaccabile")
        print("=" * 70)

        start_time = time.time()
        base_path = Path(__file__).parent

        # Esegui tutti i test cases
        for category, files in self.test_cases.items():
            print(f"\\n📁 Testing {category.upper()}:")

            for filename, expected in files.items():
                file_path = base_path / category / filename

                if not file_path.exists():
                    print(f"❌ SKIP: {filename} (file not found)")
                    continue

                result = self._validate_single_file(file_path, expected)
                self.results.append(result)

                # Stampa risultato immediato
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(
                    f"  {status} {filename} "
                    f"({result.actual_errors}/{result.expected_errors} errors, "
                    f"{result.analysis_time_ms:.1f}ms)",
                )

                if not result.success:
                    print(f"    Expected: {result.expected_types}")
                    print(f"    Actual: {result.actual_types}")

        # Calcola metriche finali
        total_time = time.time() - start_time
        metrics = self._calculate_metrics()

        # Report finale
        self._print_final_report(metrics, total_time)

        return {
            "metrics": metrics,
            "results": [self._result_to_dict(r) for r in self.results],
            "total_time_seconds": total_time,
        }

    def _validate_single_file(
        self, file_path: Path, expected: Dict[str, Any],
    ) -> ValidationResult:
        """Valida un singolo file e confronta con risultati attesi"""

        start_time = time.perf_counter()

        try:
            # Analizza il file
            analysis_result = self.analyzer.analyze_file(file_path)

            analysis_time_ms = (time.perf_counter() - start_time) * 1000

            if "error" in analysis_result:
                # Errore di parsing/analisi
                return ValidationResult(
                    file_path=str(file_path),
                    expected_errors=expected["errors"],
                    expected_types=expected["types"],
                    actual_errors=-1,  # Indica errore di sistema
                    actual_types=[],
                    success=False,
                    analysis_time_ms=analysis_time_ms,
                    details={"error": analysis_result["error"]},
                )

            # Estrai tipi di pattern rilevati
            patterns = analysis_result.get("patterns", [])
            actual_types = [p["type"] for p in patterns]
            actual_errors = len(patterns)

            # Controlla se il risultato è corretto
            success = actual_errors == expected["errors"] and set(actual_types) == set(
                expected["types"],
            )

            return ValidationResult(
                file_path=str(file_path),
                expected_errors=expected["errors"],
                expected_types=expected["types"],
                actual_errors=actual_errors,
                actual_types=actual_types,
                success=success,
                analysis_time_ms=analysis_time_ms,
                details=analysis_result,
            )

        except Exception as e:
            analysis_time_ms = (time.perf_counter() - start_time) * 1000
            return ValidationResult(
                file_path=str(file_path),
                expected_errors=expected["errors"],
                expected_types=expected["types"],
                actual_errors=-1,
                actual_types=[],
                success=False,
                analysis_time_ms=analysis_time_ms,
                details={"exception": str(e)},
            )

    def _calculate_metrics(self) -> Dict[str, float]:
        """Calcola le metriche di performance del sistema"""
        if not self.results:
            return {}

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests

        # Calcola falsi positivi e falsi negativi
        false_positives = 0  # Clean code con errori rilevati
        false_negatives = 0  # Problematic code senza errori rilevati
        true_positives = 0  # Problematic code con errori corretti
        true_negatives = 0  # Clean code senza errori

        for result in self.results:
            if result.expected_errors == 0:  # Clean code
                if result.actual_errors == 0:
                    true_negatives += 1
                else:
                    false_positives += 1
            elif result.actual_errors > 0 and result.success:
                true_positives += 1
            else:
                false_negatives += 1

        # Performance metrics
        avg_time_ms = sum(r.analysis_time_ms for r in self.results) / total_tests
        max_time_ms = max(r.analysis_time_ms for r in self.results)

        # Accuracy metrics
        accuracy = passed_tests / total_tests * 100
        false_positive_rate = (
            false_positives / total_tests * 100 if total_tests > 0 else 0
        )
        false_negative_rate = (
            false_negatives / total_tests * 100 if total_tests > 0 else 0
        )

        return {
            "accuracy_percent": accuracy,
            "false_positive_rate_percent": false_positive_rate,
            "false_negative_rate_percent": false_negative_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "avg_analysis_time_ms": avg_time_ms,
            "max_analysis_time_ms": max_time_ms,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }

    def _print_final_report(self, metrics: Dict[str, float], total_time: float):
        """Stampa il report finale con le metriche"""
        print("\\n" + "=" * 70)
        print("🏆 FINAL VALIDATION REPORT - Fase 1: Solidità Inattaccabile")
        print("=" * 70)

        # Target achievement
        accuracy = metrics["accuracy_percent"]
        fp_rate = metrics["false_positive_rate_percent"]

        print(f"📊 ACCURACY: {accuracy:.1f}% (Target: >99%)")
        print(f"📊 FALSE POSITIVES: {fp_rate:.1f}% (Target: <1%)")

        # Overall status
        if accuracy >= 99.0 and fp_rate <= 1.0:
            print("🎉 🟢 PHASE 1 TARGET ACHIEVED! Sistema pronto per produzione.")
        elif accuracy >= 95.0:
            print("🔶 🟡 GOOD PROGRESS - Needs refinement for production")
        else:
            print("🔴 ❌ CRITICAL ISSUES - Major fixes needed")

        print("\\n⚡ PERFORMANCE:")
        print(f"   Average analysis time: {metrics['avg_analysis_time_ms']:.1f}ms")
        print(f"   Max analysis time: {metrics['max_analysis_time_ms']:.1f}ms")
        print(f"   Total validation time: {total_time:.2f}s")

        print("\\n🎯 DETAILED METRICS:")
        print(f"   Tests passed: {metrics['passed_tests']}/{metrics['total_tests']}")
        print(f"   True positives: {metrics['true_positives']}")
        print(f"   True negatives: {metrics['true_negatives']}")
        print(f"   False positives: {metrics['false_positives']}")
        print(f"   False negatives: {metrics['false_negatives']}")

        # Recommendations
        print("\\n💡 NEXT STEPS:")
        if accuracy < 99.0:
            print("   - Improve pattern detection accuracy")
            print("   - Review false negative cases")
        if fp_rate > 1.0:
            print("   - Reduce false positives with contextual analysis")
            print("   - Implement global state tracking")
        if metrics["avg_analysis_time_ms"] > 300:
            print("   - Optimize analysis performance")

    def _result_to_dict(self, result: ValidationResult) -> Dict[str, Any]:
        """Converte ValidationResult in dict per JSON serialization"""
        return {
            "file_path": result.file_path,
            "expected_errors": result.expected_errors,
            "expected_types": result.expected_types,
            "actual_errors": result.actual_errors,
            "actual_types": result.actual_types,
            "success": result.success,
            "analysis_time_ms": result.analysis_time_ms,
        }


if __name__ == "__main__":
    # Crea results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Esegui validazione
    suite = ValidationSuite()
    results = suite.run_full_validation()

    # Salva risultati in JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"validation_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\\n📄 Results saved to: {results_file}")

    # Exit code basato sui risultati
    accuracy = results["metrics"]["accuracy_percent"]
    exit_code = 0 if accuracy >= 99.0 else 1
    sys.exit(exit_code)
