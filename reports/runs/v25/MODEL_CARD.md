---
license: apache-2.0
datasets:
- cardiffnlp/tweet_eval
- dair-ai/emotion
- fancyzhx/ag_news
- fancyzhx/dbpedia_14
- mteb/amazon_massive_intent
- mteb/banking77
- mteb/emotion
- mteb/imdb
- nyu-mll/glue
- stanfordnlp/sst2
model-index:
[
    {
      "name": "nanocore-s1",
      "results": [
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "CLINC150 (150 intents)",
            "type": "CLINC150 (150 intents)"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy",
              "value": 0.9044444444444445,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "CLINC150 (150 intents)",
            "type": "CLINC150 (150 intents)"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier",
              "value": 0.14626882884584277,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "CLINC150 (150 intents)",
            "type": "CLINC150 (150 intents)"
          },
          "metrics": [
            {
              "type": "coverage",
              "name": "coverage",
              "value": 0.9466666666666667,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "CLINC150 (150 intents)",
            "type": "CLINC150 (150 intents)"
          },
          "metrics": [
            {
              "type": "mean set size",
              "name": "mean set size",
              "value": 2.0155555555555558,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [cosine]",
              "value": 0.9285714285714286,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [cosine]",
              "value": 3.2722030834768114,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [cosine]",
              "value": 0.9367214532115317,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [cosine]",
              "value": 0.8892325841491605,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [cosine]",
              "value": 0.017704594814763153,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [cosine]",
              "value": 0.972936400541272,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [taskhead]",
              "value": 0.9296536796536796,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [taskhead]",
              "value": 0.5272882216235054,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [taskhead]",
              "value": 0.12351429204052118,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [taskhead]",
              "value": 0.052020701525989366,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [taskhead]",
              "value": 0.012913887896624093,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [taskhead]",
              "value": 0.9783491204330176,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [tfidf_lr]",
              "value": 0.7305194805194806,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [tfidf_lr]",
              "value": 1.8231563533503592,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [tfidf_lr]",
              "value": 0.6361228564421396,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [tfidf_lr]",
              "value": 0.4651850466665009,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [tfidf_lr]",
              "value": 0.11510065014866559,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "banking77",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [tfidf_lr]",
              "value": 0.8200270635994588,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [cosine]",
              "value": 0.4983127109111361,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [cosine]",
              "value": 1.5732585718012848,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [cosine]",
              "value": 0.749735092450099,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [cosine]",
              "value": 0.2558610087407287,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [cosine]",
              "value": 0.3435350564420917,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [cosine]",
              "value": 0.5527426160337553,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [taskhead]",
              "value": 0.625421822272216,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [taskhead]",
              "value": 1.0267521916778957,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [taskhead]",
              "value": 0.5056778729604183,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [taskhead]",
              "value": 0.059791277227860165,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [taskhead]",
              "value": 0.20842354872570165,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [taskhead]",
              "value": 0.6919831223628692,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [tfidf_lr]",
              "value": 0.5804274465691789,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [tfidf_lr]",
              "value": 1.2862711399792868,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [tfidf_lr]",
              "value": 0.6143849843900769,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [tfidf_lr]",
              "value": 0.18178945422744616,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [tfidf_lr]",
              "value": 0.2567149436938021,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "emotion",
            "type": "dair-ai/emotion",
            "config": "split",
            "split": "test",
            "revision": "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [tfidf_lr]",
              "value": 0.6540084388185654,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [cosine]",
              "value": 0.7560137457044673,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [cosine]",
              "value": 1.1769558168314214,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [cosine]",
              "value": 0.635756728505512,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [cosine]",
              "value": 0.43464105755976284,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [cosine]",
              "value": 0.08036470180529841,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [cosine]",
              "value": 0.839541547277937,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [taskhead]",
              "value": 0.7983963344788088,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [taskhead]",
              "value": 0.5633529553032346,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [taskhead]",
              "value": 0.28941590852770355,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [taskhead]",
              "value": 0.022662586663002283,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [taskhead]",
              "value": 0.06589172042027516,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [taskhead]",
              "value": 0.8782234957020058,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [tfidf_lr]",
              "value": 0.5647193585337915,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [tfidf_lr]",
              "value": 1.0543851543764506,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [tfidf_lr]",
              "value": 0.5609940322097666,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [tfidf_lr]",
              "value": 0.05078518497688804,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [tfidf_lr]",
              "value": 0.2591069670512459,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_emotion",
            "type": "cardiffnlp/tweet_eval",
            "config": "emotion",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [tfidf_lr]",
              "value": 0.6246418338108882,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [cosine]",
              "value": 0.5745745745745746,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [cosine]",
              "value": 0.9717678397283366,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [cosine]",
              "value": 0.5823935277109943,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [cosine]",
              "value": 0.14481616954621596,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [cosine]",
              "value": 0.3580024244267671,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [cosine]",
              "value": 0.5969962453066333,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [taskhead]",
              "value": 0.6656656656656657,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [taskhead]",
              "value": 0.7454962191675608,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [taskhead]",
              "value": 0.44883250592021423,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [taskhead]",
              "value": 0.06740291019064473,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [taskhead]",
              "value": 0.18266053702300217,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [taskhead]",
              "value": 0.7121401752190237,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [tfidf_lr]",
              "value": 0.48848848848848847,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [tfidf_lr]",
              "value": 1.0268892418172424,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [tfidf_lr]",
              "value": 0.6203067756591937,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [tfidf_lr]",
              "value": 0.07220854680765666,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [tfidf_lr]",
              "value": 0.4174674253684276,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_sentiment",
            "type": "cardiffnlp/tweet_eval",
            "config": "sentiment",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [tfidf_lr]",
              "value": 0.5118898623279099,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [cosine]",
              "value": 0.568,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [cosine]",
              "value": 0.674224444121813,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [cosine]",
              "value": 0.4813135160800567,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [cosine]",
              "value": 0.02964718484980378,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [cosine]",
              "value": 0.3341860977097959,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [cosine]",
              "value": 0.59125,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [taskhead]",
              "value": 0.586,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [taskhead]",
              "value": 0.7526733163873737,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [taskhead]",
              "value": 0.5279688006941894,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [taskhead]",
              "value": 0.1657468607013505,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [taskhead]",
              "value": 0.34117698505295146,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [taskhead]",
              "value": 0.61625,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [tfidf_lr]",
              "value": 0.529,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [tfidf_lr]",
              "value": 0.8583300948997423,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [tfidf_lr]",
              "value": 0.6056841708044907,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [tfidf_lr]",
              "value": 0.21910837466097147,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [tfidf_lr]",
              "value": 0.42500784829366445,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "tweet_hate",
            "type": "cardiffnlp/tweet_eval",
            "config": "hate",
            "split": "test",
            "revision": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [tfidf_lr]",
              "value": 0.54,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [cosine]",
              "value": 0.8543577981651376,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [cosine]",
              "value": 0.5288252034821672,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [cosine]",
              "value": 0.34189550278200104,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [cosine]",
              "value": 0.2390399236539277,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [cosine]",
              "value": 0.0577914802985418,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [cosine]",
              "value": 0.9025787965616046,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [taskhead]",
              "value": 0.8474770642201835,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [taskhead]",
              "value": 0.3740144519915765,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [taskhead]",
              "value": 0.22485468133733694,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [taskhead]",
              "value": 0.0460401614755893,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [taskhead]",
              "value": 0.05987504881314512,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [taskhead]",
              "value": 0.9097421203438395,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [tfidf_lr]",
              "value": 0.6823394495412844,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [tfidf_lr]",
              "value": 0.5810348548162607,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [tfidf_lr]",
              "value": 0.3982212627463629,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [tfidf_lr]",
              "value": 0.03027042893228841,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [tfidf_lr]",
              "value": 0.1957267436682309,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "sst2",
            "type": "stanfordnlp/sst2",
            "split": "validation",
            "revision": "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [tfidf_lr]",
              "value": 0.7320916905444126,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [cosine]",
              "value": 0.614,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [cosine]",
              "value": 1.3052564870529917,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [cosine]",
              "value": 0.7077980469083839,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [cosine]",
              "value": 0.3448650293288783,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [cosine]",
              "value": 0.20537757005648852,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [cosine]",
              "value": 0.7125,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [taskhead]",
              "value": 0.869,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [taskhead]",
              "value": 0.41372853092448625,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [taskhead]",
              "value": 0.21312397256507004,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [taskhead]",
              "value": 0.04417353320143835,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [taskhead]",
              "value": 0.05745808296905244,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [taskhead]",
              "value": 0.92125,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [tfidf_lr]",
              "value": 0.786,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [tfidf_lr]",
              "value": 0.6914543191732418,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [tfidf_lr]",
              "value": 0.3545972599888287,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [tfidf_lr]",
              "value": 0.16086981660147115,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [tfidf_lr]",
              "value": 0.09404806905092905,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ag_news",
            "type": "fancyzhx/ag_news",
            "split": "test",
            "revision": "eb185aade064a813bc0b7f42de02595523103ca4"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [tfidf_lr]",
              "value": 0.85625,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [cosine]",
              "value": 0.7756539235412475,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [cosine]",
              "value": 2.423271432355089,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [cosine]",
              "value": 0.8940764775999859,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [cosine]",
              "value": 0.684734975698824,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [cosine]",
              "value": 0.06740157476515518,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [cosine]",
              "value": 0.8842767295597485,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [taskhead]",
              "value": 0.9698189134808853,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [taskhead]",
              "value": 0.09611520519692308,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [taskhead]",
              "value": 0.04451038443722737,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [taskhead]",
              "value": 0.012124314962494167,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [taskhead]",
              "value": 0.005794250814196914,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [taskhead]",
              "value": 0.9974842767295597,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [tfidf_lr]",
              "value": 0.9265593561368209,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [tfidf_lr]",
              "value": 0.6002155406277121,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [tfidf_lr]",
              "value": 0.236450023237883,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [tfidf_lr]",
              "value": 0.29183704822445566,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [tfidf_lr]",
              "value": 0.011409407979434228,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "dbpedia_14",
            "type": "fancyzhx/dbpedia_14",
            "config": "dbpedia_14",
            "split": "test",
            "revision": "9abd46cf7fc8b4c64290f26993c540b92aa145ac"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [tfidf_lr]",
              "value": 0.9761006289308176,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [cosine]",
              "value": 0.3683683683683684,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [cosine]",
              "value": 1.0973625624581085,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [cosine]",
              "value": 0.6655758933365289,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [cosine]",
              "value": 0.021428152060223575,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [cosine]",
              "value": 0.608109659073268,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [cosine]",
              "value": 0.3692115143929912,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [taskhead]",
              "value": 0.4494494494494494,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [taskhead]",
              "value": 1.05138719239306,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [taskhead]",
              "value": 0.6352378242710146,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [taskhead]",
              "value": 0.03114486788283352,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [taskhead]",
              "value": 0.48329652986123717,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [taskhead]",
              "value": 0.4668335419274093,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [tfidf_lr]",
              "value": 0.3793793793793794,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [tfidf_lr]",
              "value": 1.1987966401754937,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [tfidf_lr]",
              "value": 0.7227819539691528,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [tfidf_lr]",
              "value": 0.1433931779761808,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [tfidf_lr]",
              "value": 0.6200272268418958,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "mnli",
            "type": "nyu-mll/glue",
            "config": "mnli",
            "split": "validation_matched",
            "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [tfidf_lr]",
              "value": 0.38923654568210264,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [cosine]",
              "value": 0.8877314814814815,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [cosine]",
              "value": 3.1400042535210932,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [cosine]",
              "value": 0.9286587596933781,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [cosine]",
              "value": 0.8418499831112866,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [cosine]",
              "value": 0.03215900462644963,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [cosine]",
              "value": 0.9652677279305355,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [taskhead]",
              "value": 0.8946759259259259,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [taskhead]",
              "value": 0.5984928008326266,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [taskhead]",
              "value": 0.17616032295552916,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [taskhead]",
              "value": 0.06227599392609398,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [taskhead]",
              "value": 0.02518890857091913,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [taskhead]",
              "value": 0.9638205499276411,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "accuracy [tfidf_lr]",
              "value": 0.6782407407407407,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "log_score",
              "name": "log_score [tfidf_lr]",
              "value": 1.720247422156417,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "brier",
              "name": "brier [tfidf_lr]",
              "value": 0.566834583682873,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "ece15",
              "name": "ece15 [tfidf_lr]",
              "value": 0.32096533978845054,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "aurc",
              "name": "aurc [tfidf_lr]",
              "value": 0.12057516705332816,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "massive_intent_en",
            "type": "mteb/amazon_massive_intent",
            "config": "en",
            "split": "test",
            "revision": "940fd47a81eaa7f2cc7b129674d945d618ac38c2"
          },
          "metrics": [
            {
              "type": "acc_at_80",
              "name": "acc_at_80 [tfidf_lr]",
              "value": 0.7742402315484804,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "clinc150-noul",
            "type": "clinc150",
            "split": "test"
          },
          "metrics": [
            {
              "type": "auroc",
              "name": "mean AUROC [noul, prop_in_state]",
              "value": 0.6331944444444444,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "clinc150-noul",
            "type": "clinc150",
            "split": "test"
          },
          "metrics": [
            {
              "type": "auroc",
              "name": "mean AUROC [noul, prop_in_options]",
              "value": 0.9909722222222221,
              "verified": false
            }
          ],
          "source": {
            "name": "s1_verify kernel (self-reported)"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "Banking77Classification",
            "type": "mteb/banking77",
            "split": "test",
            "revision": "0fd18e25b25c072e09e0d92ab615fda904d66300"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "main_score [encoder, MTEB harness]",
              "value": 0.9224350649350649,
              "verified": false
            }
          ],
          "source": {
            "name": "MTEB harness v2.24.2",
            "url": "https://github.com/embeddings-benchmark/mteb"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "EmotionClassification",
            "type": "mteb/emotion",
            "split": "test",
            "revision": "4f58c6b202a23cf9a4da393831edf4f9183cad37"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "main_score [encoder, MTEB harness]",
              "value": 0.5736,
              "verified": false
            }
          ],
          "source": {
            "name": "MTEB harness v2.24.2",
            "url": "https://github.com/embeddings-benchmark/mteb"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "MassiveIntentClassification",
            "type": "mteb/amazon_massive_intent",
            "split": "test",
            "revision": "4672e20407010da34463acc759c162ca9734bca6"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "main_score [encoder, MTEB harness]",
              "value": 0.880200167313305,
              "verified": false
            }
          ],
          "source": {
            "name": "MTEB harness v2.24.2",
            "url": "https://github.com/embeddings-benchmark/mteb"
          }
        },
        {
          "task": {
            "type": "text-classification"
          },
          "dataset": {
            "name": "ImdbClassification",
            "type": "mteb/imdb",
            "split": "test",
            "revision": "3d86128a09e091d6018b6d26cad27f2739fc2db7"
          },
          "metrics": [
            {
              "type": "accuracy",
              "name": "main_score [encoder, MTEB harness]",
              "value": 0.9103640000000001,
              "verified": false
            }
          ],
          "source": {
            "name": "MTEB harness v2.24.2",
            "url": "https://github.com/embeddings-benchmark/mteb"
          }
        }
      ]
    }
  ]
---

# nanocore-s1

## Model Details

**name**
nanocore-s1

**version**
add33d5+dirty

**type**
frozen multimodal encoder + trained head + conformal gate

**encoder**
google/embeddinggemma-2

**artifact digest**
207cef05b2f0a339e4a9543bbf3e3fd8c4ac9e40ec1badd603da967d05e7e92e

**license**
apache-2.0

**citation**
EmbeddingGemma 2 (Google); conformal prediction (Vovk et al.)

## Intended Use

**primary**
typed single-label decisions over a caller-supplied option set (answer / clarify / escalate / abstain) with a coverage guarantee

**users**
an application embedding a decision step; a human reviewing escalations

**out of scope**
- generation of any kind — this model emits labels, not text
- open-ended option sets it was not adapted to (it will escalate, by design)
- audio and video states: the encoder supports them, this build has never run them
- languages outside the measured set — see the language gap below
- any use where an escalation has nowhere to go

## Factors

**measured**
- language (en + de/es/fr/ru/zh-CN/ja)
- option-set size (2-150 measured)
- modality (text; vision at decision level on one task)

**not measured**
- demographic or user-attribute slices
- domain shift beyond the measured corpora
- adversarial input

## Metrics

**why**
proper scoring rules gate; ECE is reported but never gated, because sharpening minimises ECE without improving the distribution

**gated**
- log score
- Brier
- conformal coverage
- mean set size
- refusal behaviour

**reported not gated**
- ECE (15 bins)
- AURC
- selective accuracy

## Evaluation Data

**datasets**
CLINC150 (150 intents)

**splits**
fit / calibrate / evaluate, disjoint; thresholds chosen on the first two and applied unchanged to the third

**seed**
recorded in the report

## Training Data

**note**
only the head and the gate are fitted; the encoder is frozen

**source**
user-supplied labeled data

**classes**
150

**headroom**
0.7088888888888889

## Quantitative Analyses

**verdict**
qualified

**passed**
24

**failed**
- E4
- E5
- G3

**open capability gaps**
- E4
- E5

**not measured**
- F2

**criteria**
- {'id': 'A1', 'gate': 'A', 'statement': 'the contract suite is green where it runs', 'severity': 'must_pass', 'threshold': 'suite_green_on_kaggle is true', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'A2', 'gate': 'A', 'statement': 'a saved bundle decides identically after reload', 'severity': 'must_pass', 'threshold': 'bundle_roundtrip_identical is true', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'A3', 'gate': 'A', 'statement': 'the qualification names the exact artifact it qualified', 'severity': 'must_pass', 'threshold': 'provenance carries a bundle digest', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'B1', 'gate': 'B', 'statement': 'the head beats TF-IDF+LR on both proper metrics, every dataset', 'severity': 'must_pass', 'threshold': 'log and Brier strictly lower on all datasets that ran', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'B2', 'gate': 'B', 'statement': 'headroom is recorded, so a saturated task cannot be read as a win', 'severity': 'report', 'threshold': 'ADR-0011 headroom present for the primary dataset', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'B3', 'gate': 'B', 'statement': 'a shared-harness anchor is recorded for comparability', 'severity': 'report', 'threshold': 'mteb_anchor.status == ran; the card can cite leaderboard terms', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'B4', 'gate': 'B', 'statement': "the contract's non-choice qtypes produce measured evidence", 'severity': 'report', 'threshold': 'ordinal.status == ran and noul.status == ran', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'C1', 'gate': 'C', 'statement': 'cosine-path conformal coverage meets its target', 'severity': 'must_pass', 'threshold': 'coverage >= 1 - alpha', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'C2', 'gate': 'C', 'statement': 'head-path conformal coverage meets its target', 'severity': 'must_pass', 'threshold': 'coverage >= 1 - alpha', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'C3', 'gate': 'C', 'statement': 'prediction sets stay actionable on the cosine path', 'severity': 'must_pass', 'threshold': 'mean set size < 8', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'C4', 'gate': 'C', 'statement': 'prediction sets stay actionable on the head path', 'severity': 'must_pass', 'threshold': 'mean set size < 10', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'C5', 'gate': 'C', 'statement': 'the gate neither answers nor escalates everything', 'severity': 'must_pass', 'threshold': 'escalate_rate < 1.0 and something was answered', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'D1', 'gate': 'D', 'statement': 'no refusal case produced an answer', 'severity': 'must_pass', 'threshold': 'unsafe_answers == 0', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'D2', 'gate': 'D', 'statement': 'the refusal battery was not passed by refusing everything', 'severity': 'must_pass', 'threshold': 'the verbatim-option positive control answered', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'D3', 'gate': 'D', 'statement': 'answerable inputs are not declined (over-refusal)', 'severity': 'must_pass', 'threshold': 'no positive control was over-refused', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'D4', 'gate': 'D', 'statement': 'out-of-schema questions are refused loudly', 'severity': 'must_pass', 'threshold': 'noul.bundle_refuses is true — _align_scores fails loud', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'E1', 'gate': 'E', 'statement': 'the pipeline holds off-English', 'severity': 'must_pass', 'threshold': 'in-language accuracy > 0.5 for every language that ran', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'E2', 'gate': 'E', 'statement': 'an English-trained head transfers', 'severity': 'must_pass', 'threshold': 'cross-lingual accuracy > 0.5 for every language that ran', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'E3', 'gate': 'E', 'statement': 'a non-text state carries decision information', 'severity': 'must_pass', 'threshold': 'image-arm CI lower bound above chance', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'E4', 'gate': 'E', 'statement': 'two-field composition beats a single field', 'severity': 'must_fix', 'threshold': 'full beats BOTH singles on a paired CI', 'pass': False, 'status': 'fail', 'note': ''}
- {'id': 'E5', 'gate': 'E', 'statement': 'the two-field path supports entailment', 'severity': 'must_fix', 'threshold': 'mnli accuracy >= 0.60', 'pass': False, 'status': 'fail', 'note': ''}
- {'id': 'E6', 'gate': 'E', 'statement': 'the slow-state arm is not less safe than static under shift', 'severity': 'must_pass', 'threshold': "glial phase-2 wrong_answer_rate <= static's", 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'F1', 'gate': 'F', 'statement': 'a retrain can be promoted and rolled back', 'severity': 'must_pass', 'threshold': 'registry round trip with a non-empty shadow comparison', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'F2', 'gate': 'F', 'statement': 'latency is measured on the target device', 'severity': 'must_pass', 'threshold': 'Phase 8 parity run', 'pass': None, 'status': 'deferred', 'note': 'evidence absent — deferred'}
- {'id': 'G1', 'gate': 'G', 'statement': 'conformal coverage is not hiding a hard slice', 'severity': 'must_pass', 'threshold': 'no populated confidence band < target - 0.15, both legs', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'G2', 'gate': 'G', 'statement': 'deferral targets the items the model would get wrong', 'severity': 'must_pass', 'threshold': 'acc on deferred <= acc on asserted, both legs', 'pass': True, 'status': 'pass', 'note': ''}
- {'id': 'G3', 'gate': 'G', 'statement': 'abstention does not concentrate on a single intent', 'severity': 'report', 'threshold': 'no intent rejects at >5x the overall rate', 'pass': False, 'status': 'fail', 'note': ''}
- {'id': 'G4', 'gate': 'G', 'statement': 'the head is not answering from surface cues', 'severity': 'must_pass', 'threshold': 'candidate-order invariant; withheld state never answered', 'pass': True, 'status': 'pass', 'note': ''}

## Ethical Considerations

**privacy**
inputs never leave the device; the decision path opens no socket (tested), and the GGUF backend is local-only

**failure mode**
the tolerable failure is an abstention, not a confident wrong answer — refusals are gated on both sides, because a model that refuses everything is also broken (XSTest)

**bias**
not measured beyond language; state this rather than imply coverage

## Caveats

**known limitations**
- two-field / relational composition is weak (mnli below the gate floor)
- mean-pooling a multi-item state does not beat the best single item
- conformal sets are uninformative on tasks where the scorer has no signal
- no offline-to-online correlation: no deployment has been measured

**independence**
validation is currently performed by the same process that builds the model; SR 11-7 requires independence and this is the largest outstanding gap

## Evaluation results

| dataset | metric | value |
|---|---|---|
| CLINC150 (150 intents) | accuracy | 0.9044444444444445 |
| CLINC150 (150 intents) | brier | 0.14626882884584277 |
| CLINC150 (150 intents) | coverage | 0.9466666666666667 |
| CLINC150 (150 intents) | mean set size | 2.0155555555555558 |
| banking77 | accuracy [cosine] | 0.9285714285714286 |
| banking77 | log_score [cosine] | 3.2722030834768114 |
| banking77 | brier [cosine] | 0.9367214532115317 |
| banking77 | ece15 [cosine] | 0.8892325841491605 |
| banking77 | aurc [cosine] | 0.017704594814763153 |
| banking77 | acc_at_80 [cosine] | 0.972936400541272 |
| banking77 | accuracy [taskhead] | 0.9296536796536796 |
| banking77 | log_score [taskhead] | 0.5272882216235054 |
| banking77 | brier [taskhead] | 0.12351429204052118 |
| banking77 | ece15 [taskhead] | 0.052020701525989366 |
| banking77 | aurc [taskhead] | 0.012913887896624093 |
| banking77 | acc_at_80 [taskhead] | 0.9783491204330176 |
| banking77 | accuracy [tfidf_lr] | 0.7305194805194806 |
| banking77 | log_score [tfidf_lr] | 1.8231563533503592 |
| banking77 | brier [tfidf_lr] | 0.6361228564421396 |
| banking77 | ece15 [tfidf_lr] | 0.4651850466665009 |
| banking77 | aurc [tfidf_lr] | 0.11510065014866559 |
| banking77 | acc_at_80 [tfidf_lr] | 0.8200270635994588 |
| emotion | accuracy [cosine] | 0.4983127109111361 |
| emotion | log_score [cosine] | 1.5732585718012848 |
| emotion | brier [cosine] | 0.749735092450099 |
| emotion | ece15 [cosine] | 0.2558610087407287 |
| emotion | aurc [cosine] | 0.3435350564420917 |
| emotion | acc_at_80 [cosine] | 0.5527426160337553 |
| emotion | accuracy [taskhead] | 0.625421822272216 |
| emotion | log_score [taskhead] | 1.0267521916778957 |
| emotion | brier [taskhead] | 0.5056778729604183 |
| emotion | ece15 [taskhead] | 0.059791277227860165 |
| emotion | aurc [taskhead] | 0.20842354872570165 |
| emotion | acc_at_80 [taskhead] | 0.6919831223628692 |
| emotion | accuracy [tfidf_lr] | 0.5804274465691789 |
| emotion | log_score [tfidf_lr] | 1.2862711399792868 |
| emotion | brier [tfidf_lr] | 0.6143849843900769 |
| emotion | ece15 [tfidf_lr] | 0.18178945422744616 |
| emotion | aurc [tfidf_lr] | 0.2567149436938021 |
| emotion | acc_at_80 [tfidf_lr] | 0.6540084388185654 |
| tweet_emotion | accuracy [cosine] | 0.7560137457044673 |
| tweet_emotion | log_score [cosine] | 1.1769558168314214 |
| tweet_emotion | brier [cosine] | 0.635756728505512 |
| tweet_emotion | ece15 [cosine] | 0.43464105755976284 |
| tweet_emotion | aurc [cosine] | 0.08036470180529841 |
| tweet_emotion | acc_at_80 [cosine] | 0.839541547277937 |
| tweet_emotion | accuracy [taskhead] | 0.7983963344788088 |
| tweet_emotion | log_score [taskhead] | 0.5633529553032346 |
| tweet_emotion | brier [taskhead] | 0.28941590852770355 |
| tweet_emotion | ece15 [taskhead] | 0.022662586663002283 |
| tweet_emotion | aurc [taskhead] | 0.06589172042027516 |
| tweet_emotion | acc_at_80 [taskhead] | 0.8782234957020058 |
| tweet_emotion | accuracy [tfidf_lr] | 0.5647193585337915 |
| tweet_emotion | log_score [tfidf_lr] | 1.0543851543764506 |
| tweet_emotion | brier [tfidf_lr] | 0.5609940322097666 |
| tweet_emotion | ece15 [tfidf_lr] | 0.05078518497688804 |
| tweet_emotion | aurc [tfidf_lr] | 0.2591069670512459 |
| tweet_emotion | acc_at_80 [tfidf_lr] | 0.6246418338108882 |
| tweet_sentiment | accuracy [cosine] | 0.5745745745745746 |
| tweet_sentiment | log_score [cosine] | 0.9717678397283366 |
| tweet_sentiment | brier [cosine] | 0.5823935277109943 |
| tweet_sentiment | ece15 [cosine] | 0.14481616954621596 |
| tweet_sentiment | aurc [cosine] | 0.3580024244267671 |
| tweet_sentiment | acc_at_80 [cosine] | 0.5969962453066333 |
| tweet_sentiment | accuracy [taskhead] | 0.6656656656656657 |
| tweet_sentiment | log_score [taskhead] | 0.7454962191675608 |
| tweet_sentiment | brier [taskhead] | 0.44883250592021423 |
| tweet_sentiment | ece15 [taskhead] | 0.06740291019064473 |
| tweet_sentiment | aurc [taskhead] | 0.18266053702300217 |
| tweet_sentiment | acc_at_80 [taskhead] | 0.7121401752190237 |
| tweet_sentiment | accuracy [tfidf_lr] | 0.48848848848848847 |
| tweet_sentiment | log_score [tfidf_lr] | 1.0268892418172424 |
| tweet_sentiment | brier [tfidf_lr] | 0.6203067756591937 |
| tweet_sentiment | ece15 [tfidf_lr] | 0.07220854680765666 |
| tweet_sentiment | aurc [tfidf_lr] | 0.4174674253684276 |
| tweet_sentiment | acc_at_80 [tfidf_lr] | 0.5118898623279099 |
| tweet_hate | accuracy [cosine] | 0.568 |
| tweet_hate | log_score [cosine] | 0.674224444121813 |
| tweet_hate | brier [cosine] | 0.4813135160800567 |
| tweet_hate | ece15 [cosine] | 0.02964718484980378 |
| tweet_hate | aurc [cosine] | 0.3341860977097959 |
| tweet_hate | acc_at_80 [cosine] | 0.59125 |
| tweet_hate | accuracy [taskhead] | 0.586 |
| tweet_hate | log_score [taskhead] | 0.7526733163873737 |
| tweet_hate | brier [taskhead] | 0.5279688006941894 |
| tweet_hate | ece15 [taskhead] | 0.1657468607013505 |
| tweet_hate | aurc [taskhead] | 0.34117698505295146 |
| tweet_hate | acc_at_80 [taskhead] | 0.61625 |
| tweet_hate | accuracy [tfidf_lr] | 0.529 |
| tweet_hate | log_score [tfidf_lr] | 0.8583300948997423 |
| tweet_hate | brier [tfidf_lr] | 0.6056841708044907 |
| tweet_hate | ece15 [tfidf_lr] | 0.21910837466097147 |
| tweet_hate | aurc [tfidf_lr] | 0.42500784829366445 |
| tweet_hate | acc_at_80 [tfidf_lr] | 0.54 |
| sst2 | accuracy [cosine] | 0.8543577981651376 |
| sst2 | log_score [cosine] | 0.5288252034821672 |
| sst2 | brier [cosine] | 0.34189550278200104 |
| sst2 | ece15 [cosine] | 0.2390399236539277 |
| sst2 | aurc [cosine] | 0.0577914802985418 |
| sst2 | acc_at_80 [cosine] | 0.9025787965616046 |
| sst2 | accuracy [taskhead] | 0.8474770642201835 |
| sst2 | log_score [taskhead] | 0.3740144519915765 |
| sst2 | brier [taskhead] | 0.22485468133733694 |
| sst2 | ece15 [taskhead] | 0.0460401614755893 |
| sst2 | aurc [taskhead] | 0.05987504881314512 |
| sst2 | acc_at_80 [taskhead] | 0.9097421203438395 |
| sst2 | accuracy [tfidf_lr] | 0.6823394495412844 |
| sst2 | log_score [tfidf_lr] | 0.5810348548162607 |
| sst2 | brier [tfidf_lr] | 0.3982212627463629 |
| sst2 | ece15 [tfidf_lr] | 0.03027042893228841 |
| sst2 | aurc [tfidf_lr] | 0.1957267436682309 |
| sst2 | acc_at_80 [tfidf_lr] | 0.7320916905444126 |
| ag_news | accuracy [cosine] | 0.614 |
| ag_news | log_score [cosine] | 1.3052564870529917 |
| ag_news | brier [cosine] | 0.7077980469083839 |
| ag_news | ece15 [cosine] | 0.3448650293288783 |
| ag_news | aurc [cosine] | 0.20537757005648852 |
| ag_news | acc_at_80 [cosine] | 0.7125 |
| ag_news | accuracy [taskhead] | 0.869 |
| ag_news | log_score [taskhead] | 0.41372853092448625 |
| ag_news | brier [taskhead] | 0.21312397256507004 |
| ag_news | ece15 [taskhead] | 0.04417353320143835 |
| ag_news | aurc [taskhead] | 0.05745808296905244 |
| ag_news | acc_at_80 [taskhead] | 0.92125 |
| ag_news | accuracy [tfidf_lr] | 0.786 |
| ag_news | log_score [tfidf_lr] | 0.6914543191732418 |
| ag_news | brier [tfidf_lr] | 0.3545972599888287 |
| ag_news | ece15 [tfidf_lr] | 0.16086981660147115 |
| ag_news | aurc [tfidf_lr] | 0.09404806905092905 |
| ag_news | acc_at_80 [tfidf_lr] | 0.85625 |
| dbpedia_14 | accuracy [cosine] | 0.7756539235412475 |
| dbpedia_14 | log_score [cosine] | 2.423271432355089 |
| dbpedia_14 | brier [cosine] | 0.8940764775999859 |
| dbpedia_14 | ece15 [cosine] | 0.684734975698824 |
| dbpedia_14 | aurc [cosine] | 0.06740157476515518 |
| dbpedia_14 | acc_at_80 [cosine] | 0.8842767295597485 |
| dbpedia_14 | accuracy [taskhead] | 0.9698189134808853 |
| dbpedia_14 | log_score [taskhead] | 0.09611520519692308 |
| dbpedia_14 | brier [taskhead] | 0.04451038443722737 |
| dbpedia_14 | ece15 [taskhead] | 0.012124314962494167 |
| dbpedia_14 | aurc [taskhead] | 0.005794250814196914 |
| dbpedia_14 | acc_at_80 [taskhead] | 0.9974842767295597 |
| dbpedia_14 | accuracy [tfidf_lr] | 0.9265593561368209 |
| dbpedia_14 | log_score [tfidf_lr] | 0.6002155406277121 |
| dbpedia_14 | brier [tfidf_lr] | 0.236450023237883 |
| dbpedia_14 | ece15 [tfidf_lr] | 0.29183704822445566 |
| dbpedia_14 | aurc [tfidf_lr] | 0.011409407979434228 |
| dbpedia_14 | acc_at_80 [tfidf_lr] | 0.9761006289308176 |
| mnli | accuracy [cosine] | 0.3683683683683684 |
| mnli | log_score [cosine] | 1.0973625624581085 |
| mnli | brier [cosine] | 0.6655758933365289 |
| mnli | ece15 [cosine] | 0.021428152060223575 |
| mnli | aurc [cosine] | 0.608109659073268 |
| mnli | acc_at_80 [cosine] | 0.3692115143929912 |
| mnli | accuracy [taskhead] | 0.4494494494494494 |
| mnli | log_score [taskhead] | 1.05138719239306 |
| mnli | brier [taskhead] | 0.6352378242710146 |
| mnli | ece15 [taskhead] | 0.03114486788283352 |
| mnli | aurc [taskhead] | 0.48329652986123717 |
| mnli | acc_at_80 [taskhead] | 0.4668335419274093 |
| mnli | accuracy [tfidf_lr] | 0.3793793793793794 |
| mnli | log_score [tfidf_lr] | 1.1987966401754937 |
| mnli | brier [tfidf_lr] | 0.7227819539691528 |
| mnli | ece15 [tfidf_lr] | 0.1433931779761808 |
| mnli | aurc [tfidf_lr] | 0.6200272268418958 |
| mnli | acc_at_80 [tfidf_lr] | 0.38923654568210264 |
| massive_intent_en | accuracy [cosine] | 0.8877314814814815 |
| massive_intent_en | log_score [cosine] | 3.1400042535210932 |
| massive_intent_en | brier [cosine] | 0.9286587596933781 |
| massive_intent_en | ece15 [cosine] | 0.8418499831112866 |
| massive_intent_en | aurc [cosine] | 0.03215900462644963 |
| massive_intent_en | acc_at_80 [cosine] | 0.9652677279305355 |
| massive_intent_en | accuracy [taskhead] | 0.8946759259259259 |
| massive_intent_en | log_score [taskhead] | 0.5984928008326266 |
| massive_intent_en | brier [taskhead] | 0.17616032295552916 |
| massive_intent_en | ece15 [taskhead] | 0.06227599392609398 |
| massive_intent_en | aurc [taskhead] | 0.02518890857091913 |
| massive_intent_en | acc_at_80 [taskhead] | 0.9638205499276411 |
| massive_intent_en | accuracy [tfidf_lr] | 0.6782407407407407 |
| massive_intent_en | log_score [tfidf_lr] | 1.720247422156417 |
| massive_intent_en | brier [tfidf_lr] | 0.566834583682873 |
| massive_intent_en | ece15 [tfidf_lr] | 0.32096533978845054 |
| massive_intent_en | aurc [tfidf_lr] | 0.12057516705332816 |
| massive_intent_en | acc_at_80 [tfidf_lr] | 0.7742402315484804 |
| clinc150-noul | mean AUROC [noul, prop_in_state] | 0.6331944444444444 |
| clinc150-noul | mean AUROC [noul, prop_in_options] | 0.9909722222222221 |
| Banking77Classification | main_score [encoder, MTEB harness] | 0.9224350649350649 |
| EmotionClassification | main_score [encoder, MTEB harness] | 0.5736 |
| MassiveIntentClassification | main_score [encoder, MTEB harness] | 0.880200167313305 |
| ImdbClassification | main_score [encoder, MTEB harness] | 0.9103640000000001 |
