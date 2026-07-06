echo "Running IROS reconstruction pipeline for Unit-1 of the LEM-X Observatory..."
echo ""

python IROSrec_pipeline__argparsed.py --help

python IROSrec_pipeline__argparsed.py \
    --run IROSbenchmrk_detected_smoothed \
    --skyfield IROSDummy \
    --datadir baseline_2-50keV_1ks \
    --dataset detected \
    --energy_range 2.0 5.0 \
    --smoothing \
    --smoothing_snr_thresh 40 \
    --baseline_irosrec '/mnt/d/PhD_AASS/Coding/Images_fits/IROSbenchmrk_detected/' \

echo ""
echo "Finished IROS sky-field reconstruction(s)!"