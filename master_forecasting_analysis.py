#!/usr/bin/env python3
"""
Master Script for Kentucky Mesonet Forecasting Analysis
Complete workflow for RFSM and CRRL sites

Runs:
1. Feature engineering on raw mesonet data
2. Gradient boosting model training (XGBoost, LightGBM, CatBoost, Stacked)
3. Model evaluation and visualization
4. Results saving and analysis

Author: For Kentucky Mesonet ML Analysis
"""

import os
import sys
import glob
from datetime import datetime
import warnings
import contextlib
import io

# Aggressive warning and output suppression
warnings.filterwarnings('ignore')

# Silence pandas warnings about categorical conversion
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('mode.copy_on_write', True)

# Suppress specific pandas warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Redirect pandas warnings to null
import logging

logging.getLogger('pandas').setLevel(logging.ERROR)

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom classes
from mesonet_feature_engineering import MesonetFeatureEngineer
from gradient_boosting_forecasting import MesonetGradientBoostingForecaster


def main():
    """Main execution function"""

    # Context manager to suppress verbose output
    @contextlib.contextmanager
    def suppress_stdout():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

    print("=" * 80)
    print("KENTUCKY MESONET GRADIENT BOOSTING FORECASTING ANALYSIS")
    print("Predicting VT20, VT90, and VT90-VT20 difference during inversions")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # =================================================================
    # CONFIGURATION - UPDATE THESE PATHS TO YOUR DATA FILES
    # =================================================================

    # Data file paths - Updated with actual file locations
    RFSM_DATA_PATH = "/Users/cylis/Work/mes_summer25/RFSM mods/RFSM_mesonet_data_filled.csv"
    CRRL_DATA_PATH = "/Users/cylis/Work/mes_summer25/CRRL mods/CRRL_mesonet_data_filled_v2.csv"

    # Output folders
    FEATURE_FOLDER = "/Users/cylis/Work/mes_summer25/codes/feature_engineered_data"
    RESULTS_FOLDER = "/Users/cylis/Work/mes_summer25/codes/model_results"

    # Create output directories
    os.makedirs(FEATURE_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Check if data files exist
    if not os.path.exists(RFSM_DATA_PATH):
        print(f"❌ ERROR: RFSM data file not found: {RFSM_DATA_PATH}")
        print("Please update RFSM_DATA_PATH in the script to point to your actual data file")
        return

    if not os.path.exists(CRRL_DATA_PATH):
        print(f"❌ ERROR: CRRL data file not found: {CRRL_DATA_PATH}")
        print("Please update CRRL_DATA_PATH in the script to point to your actual data file")
        return

    print(f"📁 Data files found:")
    print(f"   RFSM: {RFSM_DATA_PATH}")
    print(f"   CRRL: {CRRL_DATA_PATH}")
    print(f"📁 Output folders:")
    print(f"   Features: {FEATURE_FOLDER}")
    print(f"   Results: {RESULTS_FOLDER}")
    print()

    # =================================================================
    # STEP 1: FEATURE ENGINEERING
    # =================================================================

    print("🔧 STEP 1: FEATURE ENGINEERING")
    print("-" * 50)

    # Process RFSM site
    print("🔄 Processing RFSM site...")
    try:
        fe_rfsm = MesonetFeatureEngineer(data_path=RFSM_DATA_PATH)
        rfsm_enhanced = fe_rfsm.engineer_all_features(save_to_folder=FEATURE_FOLDER)
        print("✅ RFSM feature engineering complete")

        # Get the saved file path
        rfsm_enhanced_files = glob.glob(os.path.join(FEATURE_FOLDER, "RFSM_enhanced_features_*.csv"))
        rfsm_enhanced_path = max(rfsm_enhanced_files, key=os.path.getctime) if rfsm_enhanced_files else None

    except Exception as e:
        print(f"❌ Error processing RFSM: {str(e)}")
        import traceback
        print("Full error details:")
        print(traceback.format_exc())
        print("\nTrying to continue with CRRL processing...")
        rfsm_enhanced_path = None

    # Process CRRL site
    print("🔄 Processing CRRL site...")
    try:
        fe_crrl = MesonetFeatureEngineer(data_path=CRRL_DATA_PATH)
        crrl_enhanced = fe_crrl.engineer_all_features(save_to_folder=FEATURE_FOLDER)
        print("✅ CRRL feature engineering complete")

        # Get the saved file path
        crrl_enhanced_files = glob.glob(os.path.join(FEATURE_FOLDER, "CRRL_enhanced_features_*.csv"))
        crrl_enhanced_path = max(crrl_enhanced_files, key=os.path.getctime) if crrl_enhanced_files else None

    except Exception as e:
        print(f"❌ Error processing CRRL: {str(e)}")
        import traceback
        print("Full error details:")
        print(traceback.format_exc())
        crrl_enhanced_path = None

    # Check if we have at least one successful processing
    if rfsm_enhanced_path is None and crrl_enhanced_path is None:
        print("❌ Both feature engineering steps failed. Cannot continue.")
        return
    elif rfsm_enhanced_path is None:
        print("⚠️  RFSM processing failed, continuing with CRRL only")
    elif crrl_enhanced_path is None:
        print("⚠️  CRRL processing failed, continuing with RFSM only")

    print()

    # =================================================================
    # STEP 2: GRADIENT BOOSTING MODEL TRAINING
    # =================================================================

    print("🤖 STEP 2: GRADIENT BOOSTING MODEL TRAINING")
    print("-" * 50)

    # Train models for RFSM
    print("🎯 Training models for RFSM...")
    if rfsm_enhanced_path:
        try:
            rfsm_forecaster = MesonetGradientBoostingForecaster(
                enhanced_data_path=rfsm_enhanced_path,
                site_name='RFSM'
            )

            print("Training on strong inversion periods...")
            print("   (Suppressing verbose pandas output...)")

            # Suppress verbose output during training
            with suppress_stdout():
                rfsm_results = rfsm_forecaster.train_and_evaluate_all_targets(focus_inversion=True)

            print("✅ RFSM model training complete")

            # Debug: Check what results were created
            if hasattr(rfsm_forecaster, 'results'):
                print(f"   RFSM Results keys: {list(rfsm_forecaster.results.keys())}")
                for target in rfsm_forecaster.results.keys():
                    models_in_target = [k for k in rfsm_forecaster.results[target].keys() if
                                        isinstance(rfsm_forecaster.results[target][k], dict) and 'RMSE' in
                                        rfsm_forecaster.results[target][k]]
                    print(f"   {target} models: {models_in_target}")

        except Exception as e:
            print(f"❌ Error training RFSM models: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            rfsm_forecaster = None
    else:
        print("❌ No RFSM enhanced data file found")
        rfsm_forecaster = None

    # Train models for CRRL
    print("🎯 Training models for CRRL...")
    if crrl_enhanced_path:
        try:
            crrl_forecaster = MesonetGradientBoostingForecaster(
                enhanced_data_path=crrl_enhanced_path,
                site_name='CRRL'
            )

            print("Training on strong inversion periods...")
            print("   (Suppressing verbose pandas output...)")

            # Suppress verbose output during training
            with suppress_stdout():
                crrl_results = crrl_forecaster.train_and_evaluate_all_targets(focus_inversion=True)

            print("✅ CRRL model training complete")

            # Debug: Check what results were created
            if hasattr(crrl_forecaster, 'results'):
                print(f"   CRRL Results keys: {list(crrl_forecaster.results.keys())}")
                for target in crrl_forecaster.results.keys():
                    models_in_target = [k for k in crrl_forecaster.results[target].keys() if
                                        isinstance(crrl_forecaster.results[target][k], dict) and 'RMSE' in
                                        crrl_forecaster.results[target][k]]
                    print(f"   {target} models: {models_in_target}")

        except Exception as e:
            print(f"❌ Error training CRRL models: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            crrl_forecaster = None
    else:
        print("❌ No CRRL enhanced data file found")
        crrl_forecaster = None

    print("🎯 Step 2 completed - proceeding to visualization...")
    print()

    # =================================================================
    # STEP 3: RESULTS VISUALIZATION AND ANALYSIS
    # =================================================================

    print("📊 STEP 3: RESULTS VISUALIZATION AND ANALYSIS")
    print("-" * 50)

    # Set matplotlib backend to prevent hanging
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    # Create plots directory
    plots_folder = os.path.join(RESULTS_FOLDER, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    # Plot results for RFSM
    if rfsm_forecaster:
        print("📈 Creating RFSM prediction plots...")
        try:
            # Check if results exist and have the expected structure
            if hasattr(rfsm_forecaster, 'results') and rfsm_forecaster.results:
                available_targets = list(rfsm_forecaster.results.keys())
                print(f"   Available targets: {available_targets}")

                for target in ['VT20', 'VT90', 'VT_diff']:
                    if target in available_targets:
                        print(f"   - Creating {target} predictions plot")
                        try:
                            # Create plot and save to file
                            fig = rfsm_forecaster.plot_prediction_results(target, 'Stacked',
                                                                          save_path=os.path.join(plots_folder,
                                                                                                 f"RFSM_{target}_predictions.png"))
                            plt.close('all')  # Close all figures to free memory
                            print(f"     ✅ Saved plot: RFSM_{target}_predictions.png")
                        except Exception as plot_error:
                            print(f"     ❌ Error plotting {target}: {str(plot_error)}")
                    else:
                        print(f"   - Skipping {target} (not available in results)")
            else:
                print("   ❌ No results available for RFSM")

        except Exception as e:
            print(f"   ❌ Error creating RFSM plots: {str(e)}")
            import traceback
            print(f"   Full error: {traceback.format_exc()}")

    # Plot results for CRRL
    if crrl_forecaster:
        print("📈 Creating CRRL prediction plots...")
        try:
            # Check if results exist and have the expected structure
            if hasattr(crrl_forecaster, 'results') and crrl_forecaster.results:
                available_targets = list(crrl_forecaster.results.keys())
                print(f"   Available targets: {available_targets}")

                for target in ['VT20', 'VT90', 'VT_diff']:
                    if target in available_targets:
                        print(f"   - Creating {target} predictions plot")
                        try:
                            # Create plot and save to file
                            fig = crrl_forecaster.plot_prediction_results(target, 'Stacked',
                                                                          save_path=os.path.join(plots_folder,
                                                                                                 f"CRRL_{target}_predictions.png"))
                            plt.close('all')  # Close all figures to free memory
                            print(f"     ✅ Saved plot: CRRL_{target}_predictions.png")
                        except Exception as plot_error:
                            print(f"     ❌ Error plotting {target}: {str(plot_error)}")
                    else:
                        print(f"   - Skipping {target} (not available in results)")
            else:
                print("   ❌ No results available for CRRL")

        except Exception as e:
            print(f"   ❌ Error creating CRRL plots: {str(e)}")
            import traceback
            print(f"   Full error: {traceback.format_exc()}")

    print("📊 Step 3 completed - continuing to feature importance analysis...")
    print()

    # =================================================================
    # STEP 4: FEATURE IMPORTANCE ANALYSIS
    # =================================================================

    print("🔍 STEP 4: FEATURE IMPORTANCE ANALYSIS")
    print("-" * 50)

    # Feature importance for RFSM
    if rfsm_forecaster:
        print("🔍 Analyzing RFSM feature importance...")
        try:
            if hasattr(rfsm_forecaster, 'results') and rfsm_forecaster.results:
                # Try to get feature importance for VT_diff if available
                if 'VT_diff' in rfsm_forecaster.results:
                    print("   - XGBoost feature importance for VT difference prediction")
                    importance_save_path = os.path.join(plots_folder, "RFSM_VT_diff_feature_importance.png")
                    rfsm_importance = rfsm_forecaster.get_feature_importance('VT_diff', 'XGBoost',
                                                                             save_path=importance_save_path)
                    plt.close('all')  # Close plot

                    if rfsm_importance is not None:
                        print("   - Top 5 most important features for RFSM:")
                        for i, row in rfsm_importance.head(5).iterrows():
                            print(f"     {i + 1}. {row['feature']}: {row['importance']:.4f}")
                    else:
                        print("   - Feature importance analysis failed")
                else:
                    print("   - VT_diff target not available, checking other targets...")
                    available_targets = list(rfsm_forecaster.results.keys())
                    if available_targets:
                        target = available_targets[0]
                        print(f"   - Using {target} for feature importance analysis")
                        importance_save_path = os.path.join(plots_folder, f"RFSM_{target}_feature_importance.png")
                        rfsm_importance = rfsm_forecaster.get_feature_importance(target, 'XGBoost',
                                                                                 save_path=importance_save_path)
                        plt.close('all')
            else:
                print("   - No results available for feature importance analysis")

        except Exception as e:
            print(f"   ❌ Error analyzing RFSM feature importance: {str(e)}")
            import traceback
            print(f"   Full error: {traceback.format_exc()}")

    # Feature importance for CRRL
    if crrl_forecaster:
        print("🔍 Analyzing CRRL feature importance...")
        try:
            if hasattr(crrl_forecaster, 'results') and crrl_forecaster.results:
                # Try to get feature importance for VT_diff if available
                if 'VT_diff' in crrl_forecaster.results:
                    print("   - XGBoost feature importance for VT difference prediction")
                    importance_save_path = os.path.join(plots_folder, "CRRL_VT_diff_feature_importance.png")
                    crrl_importance = crrl_forecaster.get_feature_importance('VT_diff', 'XGBoost',
                                                                             save_path=importance_save_path)
                    plt.close('all')  # Close plot

                    if crrl_importance is not None:
                        print("   - Top 5 most important features for CRRL:")
                        for i, row in crrl_importance.head(5).iterrows():
                            print(f"     {i + 1}. {row['feature']}: {row['importance']:.4f}")
                    else:
                        print("   - Feature importance analysis failed")
                else:
                    print("   - VT_diff target not available, checking other targets...")
                    available_targets = list(crrl_forecaster.results.keys())
                    if available_targets:
                        target = available_targets[0]
                        print(f"   - Using {target} for feature importance analysis")
                        importance_save_path = os.path.join(plots_folder, f"CRRL_{target}_feature_importance.png")
                        crrl_importance = crrl_forecaster.get_feature_importance(target, 'XGBoost',
                                                                                 save_path=importance_save_path)
                        plt.close('all')
            else:
                print("   - No results available for feature importance analysis")

        except Exception as e:
            print(f"   ❌ Error analyzing CRRL feature importance: {str(e)}")
            import traceback
            print(f"   Full error: {traceback.format_exc()}")

    print("🔍 Step 4 completed - proceeding to save results...")
    print()

    # =================================================================
    # STEP 5: SAVE RESULTS
    # =================================================================

    print("💾 STEP 5: SAVING RESULTS")
    print("-" * 50)

    # Save RFSM results
    if rfsm_forecaster:
        try:
            rfsm_results_path = rfsm_forecaster.save_results(RESULTS_FOLDER)
            print(f"✅ RFSM results saved: {rfsm_results_path}")
        except Exception as e:
            print(f"❌ Error saving RFSM results: {str(e)}")

    # Save CRRL results
    if crrl_forecaster:
        try:
            crrl_results_path = crrl_forecaster.save_results(RESULTS_FOLDER)
            print(f"✅ CRRL results saved: {crrl_results_path}")
        except Exception as e:
            print(f"❌ Error saving CRRL results: {str(e)}")

    # =================================================================
    # STEP 6: SUMMARY REPORT
    # =================================================================

    print("\n" + "=" * 80)
    print("📋 ANALYSIS SUMMARY REPORT")
    print("=" * 80)

    print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # RFSM Summary
    if rfsm_forecaster and hasattr(rfsm_forecaster, 'results'):
        print("🏠 RFSM SITE RESULTS:")
        try:
            for target in ['VT20', 'VT90', 'VT_diff']:
                if target in rfsm_forecaster.results:
                    best_model = min(rfsm_forecaster.results[target].keys(),
                                     key=lambda x: rfsm_forecaster.results[target][x]['RMSE']
                                     if isinstance(rfsm_forecaster.results[target][x], dict) and 'RMSE' in
                                        rfsm_forecaster.results[target][x] else float('inf'))
                    if isinstance(rfsm_forecaster.results[target][best_model], dict):
                        rmse = rfsm_forecaster.results[target][best_model]['RMSE']
                        r2 = rfsm_forecaster.results[target][best_model]['R²']
                        print(f"   {target}: Best model = {best_model}, RMSE = {rmse:.3f}, R² = {r2:.3f}")
        except:
            print("   Results available in saved files")

    # CRRL Summary
    if crrl_forecaster and hasattr(crrl_forecaster, 'results'):
        print("🏠 CRRL SITE RESULTS:")
        try:
            for target in ['VT20', 'VT90', 'VT_diff']:
                if target in crrl_forecaster.results:
                    best_model = min(crrl_forecaster.results[target].keys(),
                                     key=lambda x: crrl_forecaster.results[target][x]['RMSE']
                                     if isinstance(crrl_forecaster.results[target][x], dict) and 'RMSE' in
                                        crrl_forecaster.results[target][x] else float('inf'))
                    if isinstance(crrl_forecaster.results[target][best_model], dict):
                        rmse = crrl_forecaster.results[target][best_model]['RMSE']
                        r2 = crrl_forecaster.results[target][best_model]['R²']
                        print(f"   {target}: Best model = {best_model}, RMSE = {rmse:.3f}, R² = {r2:.3f}")
        except:
            print("   Results available in saved files")

    print()
    print("📁 OUTPUT FILES CREATED:")
    print(f"   Feature engineered data: {FEATURE_FOLDER}")
    print(f"   Model results: {RESULTS_FOLDER}")
    print()
    print("🎯 KEY FINDINGS:")
    print("   1. Strong atmospheric inversions identified (VT90 > VT20)")
    print("   2. Four gradient boosting models trained and compared")
    print("   3. Feature importance analysis reveals most predictive variables")
    print("   4. Time series forecasting performance evaluated")
    print()
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    # Configuration check
    print("🔧 CONFIGURATION CHECK")
    print("-" * 30)

    # Check if required modules can be imported
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        from sklearn.ensemble import StackingRegressor

        print("✅ All required libraries available")
    except ImportError as e:
        print(f"❌ Missing required library: {e}")
        print("Run: pip install pandas numpy matplotlib scikit-learn xgboost lightgbm catboost")
        sys.exit(1)

    # Check if our custom modules exist
    if not os.path.exists("mesonet_feature_engineering.py"):
        print("❌ mesonet_feature_engineering.py not found in current directory")
        print("Make sure you saved the feature engineering code as 'mesonet_feature_engineering.py'")
        sys.exit(1)

    if not os.path.exists("gradient_boosting_forecasting.py"):
        print("❌ gradient_boosting_forecasting.py not found in current directory")
        print("Make sure you saved the gradient boosting code as 'gradient_boosting_forecasting.py'")
        sys.exit(1)

    print("✅ All required files found")
    print()

    # Run main analysis
    main()