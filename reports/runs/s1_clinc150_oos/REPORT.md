# CLINC150 OOS — knowing-when benchmark

{
  "temperature": 0.0199526231496888,
  "qhat": 0.9975065418785215,
  "alpha": 0.1,
  "in_scope_accuracy": {
    "val": 0.7703333333333333,
    "test": 0.7562222222222222
  },
  "oos_auroc": {
    "max_sim": {
      "val": 0.9486066666666666,
      "test": 0.9335465555555555
    },
    "max_prob": {
      "val": 0.9105733333333332,
      "test": 0.8952928888888889
    },
    "margin": {
      "val": 0.80612,
      "test": 0.8066965555555555
    },
    "neg_entropy": {
      "val": 0.9295233333333333,
      "test": 0.914588
    },
    "neg_set_size": {
      "val": 0.919885,
      "test": 0.9118235555555556
    }
  },
  "selective": {
    "max_sim": [
      {
        "coverage": 0.5,
        "in_scope_acc": 0.8565121412803532
      },
      {
        "coverage": 0.6,
        "in_scope_acc": 0.8287037037037037
      },
      {
        "coverage": 0.7,
        "in_scope_acc": 0.8045173433718742
      },
      {
        "coverage": 0.8,
        "in_scope_acc": 0.7806171648987463
      },
      {
        "coverage": 0.9,
        "in_scope_acc": 0.7609921082299888
      },
      {
        "coverage": 0.95,
        "in_scope_acc": 0.7570093457943925
      },
      {
        "coverage": 1.0,
        "in_scope_acc": 0.7562222222222222
      }
    ],
    "max_prob": [
      {
        "coverage": 0.5,
        "in_scope_acc": 0.9095645701525865
      },
      {
        "coverage": 0.6,
        "in_scope_acc": 0.8766050735984967
      },
      {
        "coverage": 0.7,
        "in_scope_acc": 0.8388677191072401
      },
      {
        "coverage": 0.8,
        "in_scope_acc": 0.8051533742331288
      },
      {
        "coverage": 0.9,
        "in_scope_acc": 0.7714481811942348
      },
      {
        "coverage": 0.95,
        "in_scope_acc": 0.7595616193245359
      },
      {
        "coverage": 1.0,
        "in_scope_acc": 0.7562222222222222
      }
    ],
    "margin": [
      {
        "coverage": 0.5,
        "in_scope_acc": 0.9184679560106181
      },
      {
        "coverage": 0.6,
        "in_scope_acc": 0.8867376573088093
      },
      {
        "coverage": 0.7,
        "in_scope_acc": 0.8526556253569388
      },
      {
        "coverage": 0.8,
        "in_scope_acc": 0.8154241645244216
      },
      {
        "coverage": 0.9,
        "in_scope_acc": 0.786628733997155
      },
      {
        "coverage": 0.95,
        "in_scope_acc": 0.7700412276683463
      },
      {
        "coverage": 1.0,
        "in_scope_acc": 0.7562222222222222
      }
    ],
    "neg_entropy": [
      {
        "coverage": 0.5,
        "in_scope_acc": 0.8921713441654358
      },
      {
        "coverage": 0.6,
        "in_scope_acc": 0.8584729981378026
      },
      {
        "coverage": 0.7,
        "in_scope_acc": 0.8290968090859925
      },
      {
        "coverage": 0.8,
        "in_scope_acc": 0.7950401167031363
      },
      {
        "coverage": 0.9,
        "in_scope_acc": 0.7665909090909091
      },
      {
        "coverage": 0.95,
        "in_scope_acc": 0.7588826815642458
      },
      {
        "coverage": 1.0,
        "in_scope_acc": 0.7562222222222222
      }
    ],
    "neg_set_size": [
      {
        "coverage": 0.5,
        "in_scope_acc": 0.852627710400588
      },
      {
        "coverage": 0.6,
        "in_scope_acc": 0.8380716934487021
      },
      {
        "coverage": 0.7,
        "in_scope_acc": 0.8188720173535792
      },
      {
        "coverage": 0.8,
        "in_scope_acc": 0.7871613375640713
      },
      {
        "coverage": 0.9,
        "in_scope_acc": 0.7669190181234228
      },
      {
        "coverage": 0.95,
        "in_scope_acc": 0.7588222072375814
      },
      {
        "coverage": 1.0,
        "in_scope_acc": 0.7562222222222222
      }
    ]
  },
  "clarify": {
    "in_scope_with_small_set_frac": 0.15509090909090908,
    "true_intent_in_small_set": 0.9964830011723329,
    "mean_set_size_in_scope": 56.73688888888889,
    "mean_set_size_oos": 123.839,
    "coverage_at_alpha": 0.9968888888888889
  },
  "headroom_check": "n/a \u2014 this is a detection benchmark, not a head comparison",
  "verdict": {
    "oos_separable": true,
    "coverage_holds": false,
    "clarify_resolves": true
  }
}