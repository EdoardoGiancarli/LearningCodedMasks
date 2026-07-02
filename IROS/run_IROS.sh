echo "Running IROS reconstruction pipeline for Unit-1 of the LEM-X Observatory..."
echo ""

python IROSrec_pipeline__argparsed.py --help

python IROSrec_pipeline__argparsed.py \
    --run IROSbenchmrk \
    --skyfield IROSDummy \
    --datadir baseline_2-50keV_1ks \

python IROSrec_pipeline__argparsed.py \
    --run IROSbenchmrk_detected \
    --skyfield IROSDummy \
    --datadir baseline_2-50keV_1ks \
    --dataset detected \

python IROSrec_pipeline__argparsed.py \
    --run IROSbenchmrk_detected_2-5keV \
    --skyfield IROSDummy \
    --datadir baseline_2-50keV_1ks \
    --dataset detected \
    --energy_range 2.0 5.0 \

echo ""
echo "Finished IROS sky-field reconstruction(s)!"