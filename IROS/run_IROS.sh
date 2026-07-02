echo "Running IROS reconstruction pipeline for Unit-1 of the LEM-X Observatory..."
echo ""

python IROSrec_pipeline__argparsed.py --help

python IROSrec_pipeline__argparsed.py \
    --run GC_rec_detected \
    --skyfield GalacticCentre \
    --datadir baseline_2-50keV_1ks \
    --dataset detected \

python IROSrec_pipeline__argparsed.py \
    --run GC_rec_detected_2-6keV \
    --skyfield GalacticCentre \
    --datadir baseline_2-50keV_1ks \
    --dataset detected \
    --energy_range 2.0 6.0 \

echo ""
echo "Finished IROS sky-field reconstruction(s)!"