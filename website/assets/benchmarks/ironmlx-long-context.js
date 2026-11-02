window.IRONMLX_BENCHMARKS = window.IRONMLX_BENCHMARKS || {};
window.IRONMLX_BENCHMARKS["ironmlx-long-context"] = {
 "id": "ironmlx-long-context",
 "mode": "scaling",
 "app": "IronMLX 0.2.0",
 "run_utc": [
  "2026-11-01T15:41:59Z",
  "2026-11-01T20:39:33Z"
 ],
 "conditions": "128K and 256K prompts, IronMLX only; cooled to macOS thermal pressure 0 before every session; 30K points from the published B1 30K comparison runs",
 "source": [
  {
   "suite": "ironmlx-020-dense-b1-128k-thermal",
   "analysis_sha256": "5221a9e942571c9aab6f30abcbe8a1bfa166138ddeed05fe4df1ae4e2a884296",
   "run_sha256": "91d85484ee8b3d41a46777255bcd7bab106ae5782013c8d2220475e35b0edc6a"
  },
  {
   "suite": "ironmlx-020-dense-b1-256k-thermal",
   "analysis_sha256": "d472bfdb052ca78120ec24905ffd5ac05d0c93de3132690e6f4d102e9568d5e5",
   "run_sha256": "353ec545aeff20e0c1379bfceaf1f449508b241745f67aabc276463891c2f336"
  },
  {
   "suite": "ironmlx-020-moe-b1-128k-thermal",
   "analysis_sha256": "9ba9aaae43be3b28fb04681cf4e6fa5323d6265fcc678e6755476b58b422539c",
   "run_sha256": "72f1fe890e438652559255b1b06516b3f3ac1813186691ba01368dcd59bb602a"
  },
  {
   "suite": "ironmlx-020-moe-b1-256k-thermal",
   "analysis_sha256": "8ff7572c6d96656e9af6966ecc3f5418b08cd0c184633f26da52090e3a7768ab",
   "run_sha256": "77bd854d649edd617b97c34ff37e2058063e36548587bf114716d230261592f0"
  }
 ],
 "models": [
  {
   "key": "dense",
   "name": "Qwen3.8 27B",
   "points": [
    {
     "prompt_tokens": 30732,
     "length": "30K",
     "reference": "qwen38-dense-b1-30k",
     "rounds": 4,
     "prefill_tps": {
      "median": 861.9734123815383,
      "range": [
       849.6055938777813,
       862.4534264773574
      ]
     },
     "ttft_s": {
      "median": 35.83949380228296,
      "range": [
       35.81585139594972,
       36.38992810435593
      ]
     },
     "decode_tps": {
      "median": 66.48337582288981,
      "range": [
       66.08248061562239,
       66.68274638516156
      ]
     },
     "e2e_s": {
      "median": 41.04922606260516,
      "range": [
       41.003098875284195,
       41.38415641710162
      ]
     },
     "memory_gib": {
      "median": 20.58723508194089,
      "range": [
       20.52512412518263,
       20.987953692674637
      ]
     }
    },
    {
     "prompt_tokens": 131084,
     "length": "128K",
     "rounds": 2,
     "prefill_tps": {
      "median": 481.83392319170184,
      "range": [
       480.22906191581114,
       483.4387844675926
      ]
     },
     "ttft_s": {
      "median": 272.149538656231,
      "range": [
       271.24935666657984,
       273.0497206458822
      ]
     },
     "decode_tps": {
      "median": 41.96402699005558,
      "range": [
       41.58098133037075,
       42.34707264974041
      ]
     },
     "e2e_s": {
      "median": 279.8528219170403,
      "range": [
       279.1279780836776,
       280.57766575040296
      ]
     },
     "memory_gib": {
      "median": 26.82943008840084,
      "range": [
       26.82871290296316,
       26.83014727383852
      ]
     }
    },
    {
     "prompt_tokens": 261132,
     "length": "256K",
     "rounds": 2,
     "prefill_tps": {
      "median": 325.52460678134565,
      "range": [
       325.0778760360288,
       325.9713375266625
      ]
     },
     "ttft_s": {
      "median": 802.3443947294727,
      "range": [
       801.2408106252551,
       803.4479788336903
      ]
     },
     "decode_tps": {
      "median": 28.874565092251483,
      "range": [
       28.86066539509126,
       28.888464789411707
      ]
     },
     "e2e_s": {
      "median": 814.2982731249649,
      "range": [
       810.8754016454332,
       817.7211446044967
      ]
     },
     "memory_gib": {
      "median": 35.29181720688939,
      "range": [
       35.23920513689518,
       35.3444292768836
      ]
     }
    }
   ]
  },
  {
   "key": "moe",
   "name": "Qwen3.6 35B A3B",
   "points": [
    {
     "prompt_tokens": 30732,
     "length": "30K",
     "reference": "qwen36-moe-b1-30k",
     "rounds": 4,
     "prefill_tps": {
      "median": 4776.182901234617,
      "range": [
       4729.908399525993,
       4784.300462492545
      ]
     },
     "ttft_s": {
      "median": 6.469958822475746,
      "range": [
       6.457959895487875,
       6.5274113533087075
      ]
     },
     "decode_tps": {
      "median": 189.3323862160638,
      "range": [
       188.92817769594524,
       189.53441740865838
      ]
     },
     "e2e_s": {
      "median": 7.888630895642564,
      "range": [
       7.8623430002480745,
       7.955063708126545
      ]
     },
     "memory_gib": {
      "median": 21.97225596010685,
      "range": [
       21.936725713312626,
       22.00195723772049
      ]
     }
    },
    {
     "prompt_tokens": 131084,
     "length": "128K",
     "rounds": 2,
     "prefill_tps": {
      "median": 1904.4979002831403,
      "range": [
       1902.3964530661815,
       1906.5993475000992
      ]
     },
     "ttft_s": {
      "median": 68.93127034371719,
      "range": [
       68.8548931456171,
       69.00764754181728
      ]
     },
     "decode_tps": {
      "median": 143.18546252822387,
      "range": [
       142.7503881527112,
       143.62053690373654
      ]
     },
     "e2e_s": {
      "median": 70.82001070817932,
      "range": [
       70.74641037452966,
       70.89361104182899
      ]
     },
     "memory_gib": {
      "median": 23.873417243361473,
      "range": [
       23.81991995126009,
       23.926914535462856
      ]
     }
    },
    {
     "prompt_tokens": 261132,
     "length": "256K",
     "rounds": 2,
     "prefill_tps": {
      "median": 1071.177160369093,
      "range": [
       1070.4166775639374,
       1071.9376431742485
      ]
     },
     "ttft_s": {
      "median": 243.9430644372478,
      "range": [
       243.77177883312106,
       244.11435004137456
      ]
     },
     "decode_tps": {
      "median": 100.35397980419037,
      "range": [
       100.05297216066532,
       100.65498744771543
      ]
     },
     "e2e_s": {
      "median": 247.15797248994932,
      "range": [
       247.08119572978467,
       247.23474925011396
      ]
     },
     "memory_gib": {
      "median": 26.45630295202136,
      "range": [
       26.358600914478302,
       26.55400498956442
      ]
     }
    }
   ]
  }
 ]
};
