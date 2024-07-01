# wasspy

```bash
wasspy/
├── README.md
├── setup.py
├── src
│   ├── create_wass_nc_batch.py
│   ├── devel_scripts
│   ├── gen_extrinsics_fromSubSet_batch.py
│   ├── run_fullwasspipeline_batch.py
│   └── run_wass_withMYextrinsics_batch.py
├── tests
│   ├── config
│   │   ├── distortion_00.xml
│   │   ├── distortion_01.xml
│   │   ├── ext_R.xml
│   │   ├── ext_T.xml
│   │   ├── intrinsics_00.xml
│   │   ├── intrinsics_01.xml
│   │   ├── matcher_config.txt
│   │   └── stereo_config.txt
│   ├── extrinsic_calibration_test
│   │   ├── 000000_wd
│   │   ├── 000002_wd
│   │   ├── 000004_wd
│   │   ├── 000006_wd
│   │   ├── median_ext_R.xml
│   │   ├── median_ext_T.xml
│   │   ├── plots_ext_R
│   │   ├── plots_ext_T
│   │   └── workspaces.txt
│   ├── gridding
│   ├── input
│   │   ├── cam1
│   │   └── cam2
│   ├── output
│   │   ├── 000000_wd
│   │   ├── 000001_wd
│   │   ├── 000002_wd
│   │   ├── all_planes.txt
│   │   ├── figs
│   │   ├── median_plane.txt
│   │   └── workspaces.txt
│   ├── output_full
│   └── test_wasspy.py
└── wasspy.py
