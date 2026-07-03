echo "Running IROS reconstruction pipeline for Unit-1 of the LEM-X Observatory..."
echo ""

python IROSrec_pipeline__argparsed.py --help

python IROSrec_pipeline__argparsed.py \
    --run IROSbenchmrk___oldVignetting \
    --skyfield IROSDummy \
    --datadir baseline_2-50keV_1ks \
    --max_iters 15 \

python IROSrec_pipeline__argparsed.py \
    --run IROSbenchmrk_detected___oldVignetting \
    --skyfield IROSDummy \
    --datadir baseline_2-50keV_1ks \
    --dataset detected \
    --max_iters 15 \

python IROSrec_pipeline__argparsed.py \
    --run IROSbenchmrk_detected_2-5keV___oldVignetting \
    --skyfield IROSDummy \
    --datadir baseline_2-50keV_1ks \
    --dataset detected \
    --energy_range 2.0 5.0 \
    --max_iters 15 \

echo ""
echo "Finished IROS sky-field reconstruction(s)!"