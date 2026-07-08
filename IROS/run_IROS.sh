echo "Running IROS reconstruction pipeline for Unit-1 of the LEM-X Observatory..."
echo ""

python IROSrec_pipeline__argparsed.py --help

python IROSrec_pipeline__argparsed.py \
   --run GC_upx5upy1_detected_2-6keV_noVignetting \
   --skyfield GalacticCentre \
   --datadir baseline_2-50keV_1ks \
   --dataset detected \
   --up_fine 5 \
   --energy_range 2.0 6.0 \
   --thin_mask \

python IROSrec_pipeline__argparsed.py \
   --run GC_upx5upy1_detected_2-6keV \
   --skyfield GalacticCentre \
   --datadir baseline_2-50keV_1ks \
   --dataset detected \
   --up_fine 5 \
   --energy_range 2.0 6.0 \

python IROSrec_pipeline__argparsed.py \
   --run GC_upx5upy1_detected \
   --skyfield GalacticCentre \
   --datadir baseline_2-50keV_1ks \
   --dataset detected \
   --up_fine 5 \

python IROSrec_pipeline__argparsed.py \
   --run GC_upx5upy1_2-6keV \
   --skyfield GalacticCentre \
   --datadir baseline_2-50keV_1ks \
   --up_fine 5 \
   --energy_range 2.0 6.0 \

python IROSrec_pipeline__argparsed.py \
   --run GC_upx5upy1 \
   --skyfield GalacticCentre \
   --datadir baseline_2-50keV_1ks \
   --up_fine 5 \

#python IROSrec_pipeline__argparsed.py \
#    --run IROSbenchmrk_detected_smoothed \
#    --skyfield IROSDummy \
#    --datadir baseline_2-50keV_1ks \
#    --dataset detected \
#    --energy_range 2.0 5.0 \
#    --smoothing \
#    --smoothing_snr_thresh 40 \
#    --baseline_irosrec '/mnt/d/PhD_AASS/Coding/Images_fits/IROSbenchmrk_detected/' \

echo ""
echo "Finished IROS sky-field reconstruction(s)!"