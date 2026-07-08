echo "Running IROS reconstruction pipeline for Unit-1 of the LEM-X Observatory..."
echo ""

python IROSrec_pipeline__argparsed.py --help

python IROSrec_pipeline__argparsed.py \
   --run GC_upx5upy1_detected_2-6keV_noSCOX1 \
   --skyfield GalacticCentre \
   --datadir baseline_2-50keV_1ks \
   --dataset detected \
   --up_fine 5 \
   --energy_range 2.0 6.0 \
   --photons_coords 244.979705810547 -15.6400995254517 \

python IROSrec_pipeline__argparsed.py \
   --run IROSbenchmrk_upx5upy1_detected_2-5keV_noS17 \
   --skyfield IROSDummy \
   --datadir baseline_2-50keV_1ks \
   --dataset detected \
   --up_fine 5 \
   --energy_range 2.0 5.0 \
   --photons_coords 239.824508666992 -54.0559692382813 \

echo ""
echo "Finished IROS sky-field reconstruction(s)!"