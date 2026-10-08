# Ambiguity-aware policy v2 — corrected triggers

{
  "alpha": 0.1,
  "calibration": {
    "cosine": {
      "t_prob": 0.25,
      "tau_answer": 0.015279530048182147,
      "qhat": 0.023939330130815506,
      "t_set": 1.0,
      "val_a_answer_coverage": 0.4012903225806452
    },
    "taskhead": {
      "t_prob": 0.9758064516129032,
      "tau_answer": 0.2522914225541262,
      "qhat": 0.9999356269836426,
      "t_set": 1.0,
      "val_a_answer_coverage": 0.9658064516129032
    }
  },
  "tau_in_schema": 0.6991039514541626,
  "in_schema": {
    "oos_caught": 0.63,
    "in_scope_false_reject": 0.03911111111111111
  },
  "policies": {
    "always_answer": {
      "actions": {
        "answer": 5500
      },
      "resolved": 0.6187272727272727,
      "encodes_per_item": 1.0
    },
    "threshold": {
      "actions": {
        "abstain/escalate": 806,
        "answer": 4694
      },
      "resolved": 0.7185454545454546,
      "encodes": 5500,
      "coverage_hits": 3403,
      "n_in": 4500,
      "encodes_per_item": 1.0,
      "set_coverage_in_scope": 0.7562222222222222
    },
    "ambiguity_v2": {
      "actions": {
        "escalate": 3648,
        "answer": 1797,
        "clarify": 55
      },
      "resolved": 0.48163636363636364,
      "encodes": 5555,
      "coverage_hits": 4113,
      "n_in": 4500,
      "encodes_per_item": 1.01,
      "set_coverage_in_scope": 0.914
    },
    "ambiguity_v2_head": {
      "actions": {
        "answer": 4342,
        "escalate": 1158
      },
      "resolved": 0.9178181818181819,
      "encodes": 5500,
      "coverage_hits": 4498,
      "n_in": 4500,
      "encodes_per_item": 1.0,
      "set_coverage_in_scope": 0.9995555555555555
    }
  },
  "taskhead": {
    "in_scope_accuracy": 0.9704444444444444,
    "tau": 0.8644477725028992,
    "in_schema": {
      "oos_caught": 0.869,
      "in_scope_false_reject": 0.06422222222222222
    }
  },
  "verdict": {
    "v2_beats_threshold": false,
    "v2_beats_base": false,
    "coverage_holds": true,
    "deltas": {
      "always_answer": -0.0998,
      "ambiguity_v2": -0.2369,
      "ambiguity_v2_head": 0.1993
    }
  }
}