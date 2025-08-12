When tweaking your topic model, the number of topics that are generated has a large effect on the quality of the topic representations. Some topics could be merged and having an understanding of the effect will help you understand which topics should and which should not be merged.

That is where hierarchical topic modeling comes in. It tries to model the possible hierarchical nature of the topics you have created to understand which topics are similar to each other. Moreover, you will have more insight into sub-topics that might exist in your data.

<br>
<div class="svg_image">
--8<-- "docs/getting_started/hierarchicaltopics/hierarchical.svg"
</div>
<br>

In BERTopic, we can approximate this potential hierarchy by making use of our topic-term matrix (c-TF-IDF matrix). This matrix contains information about the importance of every word in every topic and makes for a nice numerical representation of our topics. The smaller the distance between two c-TF-IDF representations, the more similar we assume they are. In practice, this process of merging topics is done through the hierarchical clustering capabilities of `scipy` (see [here](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)). It allows for several linkage methods through which we can approximate our topic hierarchy. As a default, we are using the [ward](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.ward.html#scipy.cluster.hierarchy.ward) but many others are available.

Whenever we merge two topics, we can calculate the c-TF-IDF representation of these two merged by summing their bag-of-words representation. We assume that two sets of topics are merged and that all others are kept the same, regardless of their location in the hierarchy. This helps us isolate the potential effect of merging sets of topics. As a result, we can see the topic representation at each level in the tree.

## **Example**
To demonstrate hierarchical topic modeling with BERTopic, we use the 20 Newsgroups dataset to see how the topics that we uncover are represented in the 20 categories of documents.

First, we train a basic BERTopic model:

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))["data"]
topic_model = BERTopic(verbose=True)
topics, probs = topic_model.fit_transform(docs)
```

Next, we can use our fitted BERTopic model to extract possible hierarchies from our c-TF-IDF matrix:

```python
hierarchical_topics = topic_model.hierarchical_topics(docs)
```

The resulting `hierarchical_topics` is a dataframe in which merged topics are described. For example, if you would
merge two topics, what would the topic representation of the new topic be?

## **Linkage functions**

When creating the potential hierarchical nature of topics, we use Scipy's ward `linkage` function as a default
to generate the hierarchy. However, you might want to use a [different linkage function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html)
for your use case, such as `single`, `complete`, `average`, `centroid`, or `median`. In BERTopic, you can define the
linkage function yourself, including the distance function that you would like to use:


```python
from scipy.cluster import hierarchy as sch
from bertopic import BERTopic
topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)

# Hierarchical topics
linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
hierarchical_topics = topic_model.hierarchical_topics(docs, linkage_function=linkage_function)
```


## **Visualizations**
To visualize these results, we can start by running a familiar function, namely `topic_model.visualize_hierarchy`:

```python
topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
```
<iframe src="hierarchical_topics.html" style="width:1000px; height: 2150px; border: 0px;""></iframe>

If you **hover** over the black circles, you will see the topic representation at that level of the hierarchy. These representations
help you understand the effect of merging certain topics. Some might be logical to merge whilst others might not. Moreover,
we can now see which sub-topics can be found within certain larger themes.

Although this gives a nice overview of the potential hierarchy, hovering over all black circles can be tiresome. Instead, we can
use `topic_model.get_topic_tree` to create a text-based representation of this hierarchy. Although the general structure is more difficult
to view, we can see better which topics could be logically merged:

```python
>>> tree = topic_model.get_topic_tree(hierarchical_topics)
>>> print(tree)
.
└─atheists_atheism_god_moral_atheist
     ├─atheists_atheism_god_atheist_argument
     │    ├─■──atheists_atheism_god_atheist_argument ── Topic: 21
     │    └─■──br_god_exist_genetic_existence ── Topic: 124
     └─■──moral_morality_objective_immoral_morals ── Topic: 29
```

<details>
  <summary>Click here to view the full tree.</summary>

  ```bash
    .
    ├─people_armenian_said_god_armenians
    │    ├─god_jesus_jehovah_lord_christ
    │    │    ├─god_jesus_jehovah_lord_christ
    │    │    │    ├─jehovah_lord_mormon_mcconkie_god
    │    │    │    │    ├─■──ra_satan_thou_god_lucifer ── Topic: 94
    │    │    │    │    └─■──jehovah_lord_mormon_mcconkie_unto ── Topic: 78
    │    │    │    └─jesus_mary_god_hell_sin
    │    │    │         ├─jesus_hell_god_eternal_heaven
    │    │    │         │    ├─hell_jesus_eternal_god_heaven
    │    │    │         │    │    ├─■──jesus_tomb_disciples_resurrection_john ── Topic: 69
    │    │    │         │    │    └─■──hell_eternal_god_jesus_heaven ── Topic: 53
    │    │    │         │    └─■──aaron_baptism_sin_law_god ── Topic: 89
    │    │    │         └─■──mary_sin_maria_priest_conception ── Topic: 56
    │    │    └─■──marriage_married_marry_ceremony_marriages ── Topic: 110
    │    └─people_armenian_armenians_said_mr
    │         ├─people_armenian_armenians_said_israel
    │         │    ├─god_homosexual_homosexuality_atheists_sex
    │         │    │    ├─homosexual_homosexuality_sex_gay_homosexuals
    │         │    │    │    ├─■──kinsey_sex_gay_men_sexual ── Topic: 44
    │         │    │    │    └─homosexuality_homosexual_sin_homosexuals_gay
    │         │    │    │         ├─■──gay_homosexual_homosexuals_sexual_cramer ── Topic: 50
    │         │    │    │         └─■──homosexuality_homosexual_sin_paul_sex ── Topic: 27
    │         │    │    └─god_atheists_atheism_moral_atheist
    │         │    │         ├─islam_quran_judas_islamic_book
    │         │    │         │    ├─■──jim_context_challenges_articles_quote ── Topic: 36
    │         │    │         │    └─islam_quran_judas_islamic_book
    │         │    │         │         ├─■──islam_quran_islamic_rushdie_muslims ── Topic: 31
    │         │    │         │         └─■──judas_scripture_bible_books_greek ── Topic: 33
    │         │    │         └─atheists_atheism_god_moral_atheist
    │         │    │              ├─atheists_atheism_god_atheist_argument
    │         │    │              │    ├─■──atheists_atheism_god_atheist_argument ── Topic: 21
    │         │    │              │    └─■──br_god_exist_genetic_existence ── Topic: 124
    │         │    │              └─■──moral_morality_objective_immoral_morals ── Topic: 29
    │         │    └─armenian_armenians_people_israel_said
    │         │         ├─armenian_armenians_israel_people_jews
    │         │         │    ├─tax_rights_government_income_taxes
    │         │         │    │    ├─■──rights_right_slavery_slaves_residence ── Topic: 106
    │         │         │    │    └─tax_government_taxes_income_libertarians
    │         │         │    │         ├─■──government_libertarians_libertarian_regulation_party ── Topic: 58
    │         │         │    │         └─■──tax_taxes_income_billion_deficit ── Topic: 41
    │         │         │    └─armenian_armenians_israel_people_jews
    │         │         │         ├─gun_guns_militia_firearms_amendment
    │         │         │         │    ├─■──blacks_penalty_death_cruel_punishment ── Topic: 55
    │         │         │         │    └─■──gun_guns_militia_firearms_amendment ── Topic: 7
    │         │         │         └─armenian_armenians_israel_jews_turkish
    │         │         │              ├─■──israel_israeli_jews_arab_jewish ── Topic: 4
    │         │         │              └─■──armenian_armenians_turkish_armenia_azerbaijan ── Topic: 15
    │         │         └─stephanopoulos_president_mr_myers_ms
    │         │              ├─■──serbs_muslims_stephanopoulos_mr_bosnia ── Topic: 35
    │         │              └─■──myers_stephanopoulos_president_ms_mr ── Topic: 87
    │         └─batf_fbi_koresh_compound_gas
    │              ├─■──reno_workers_janet_clinton_waco ── Topic: 77
    │              └─batf_fbi_koresh_gas_compound
    │                   ├─batf_koresh_fbi_warrant_compound
    │                   │    ├─■──batf_warrant_raid_compound_fbi ── Topic: 42
    │                   │    └─■──koresh_batf_fbi_children_compound ── Topic: 61
    │                   └─■──fbi_gas_tear_bds_building ── Topic: 23
    └─use_like_just_dont_new
        ├─game_team_year_games_like
        │    ├─game_team_games_25_year
        │    │    ├─game_team_games_25_season
        │    │    │    ├─window_printer_use_problem_mhz
        │    │    │    │    ├─mhz_wire_simms_wiring_battery
        │    │    │    │    │    ├─simms_mhz_battery_cpu_heat
        │    │    │    │    │    │    ├─simms_pds_simm_vram_lc
        │    │    │    │    │    │    │    ├─■──pds_nubus_lc_slot_card ── Topic: 119
        │    │    │    │    │    │    │    └─■──simms_simm_vram_meg_dram ── Topic: 32
        │    │    │    │    │    │    └─mhz_battery_cpu_heat_speed
        │    │    │    │    │    │         ├─mhz_cpu_speed_heat_fan
        │    │    │    │    │    │         │    ├─mhz_cpu_speed_heat_fan
        │    │    │    │    │    │         │    │    ├─■──fan_cpu_heat_sink_fans ── Topic: 92
        │    │    │    │    │    │         │    │    └─■──mhz_speed_cpu_fpu_clock ── Topic: 22
        │    │    │    │    │    │         │    └─■──monitor_turn_power_computer_electricity ── Topic: 91
        │    │    │    │    │    │         └─battery_batteries_concrete_duo_discharge
        │    │    │    │    │    │              ├─■──duo_battery_apple_230_problem ── Topic: 121
        │    │    │    │    │    │              └─■──battery_batteries_concrete_discharge_temperature ── Topic: 75
        │    │    │    │    │    └─wire_wiring_ground_neutral_outlets
        │    │    │    │    │         ├─wire_wiring_ground_neutral_outlets
        │    │    │    │    │         │    ├─wire_wiring_ground_neutral_outlets
        │    │    │    │    │         │    │    ├─■──leds_uv_blue_light_boards ── Topic: 66
        │    │    │    │    │         │    │    └─■──wire_wiring_ground_neutral_outlets ── Topic: 120
        │    │    │    │    │         │    └─scope_scopes_phone_dial_number
        │    │    │    │    │         │         ├─■──dial_number_phone_line_output ── Topic: 93
        │    │    │    │    │         │         └─■──scope_scopes_motorola_generator_oscilloscope ── Topic: 113
        │    │    │    │    │         └─celp_dsp_sampling_antenna_digital
        │    │    │    │    │              ├─■──antenna_antennas_receiver_cable_transmitter ── Topic: 70
        │    │    │    │    │              └─■──celp_dsp_sampling_speech_voice ── Topic: 52
        │    │    │    │    └─window_printer_xv_mouse_windows
        │    │    │    │         ├─window_xv_error_widget_problem
        │    │    │    │         │    ├─error_symbol_undefined_xterm_rx
        │    │    │    │         │    │    ├─■──symbol_error_undefined_doug_parse ── Topic: 63
        │    │    │    │         │    │    └─■──rx_remote_server_xdm_xterm ── Topic: 45
        │    │    │    │         │    └─window_xv_widget_application_expose
        │    │    │    │         │         ├─window_widget_expose_application_event
        │    │    │    │         │         │    ├─■──gc_mydisplay_draw_gxxor_drawing ── Topic: 103
        │    │    │    │         │         │    └─■──window_widget_application_expose_event ── Topic: 25
        │    │    │    │         │         └─xv_den_polygon_points_algorithm
        │    │    │    │         │              ├─■──den_polygon_points_algorithm_polygons ── Topic: 28
        │    │    │    │         │              └─■──xv_24bit_image_bit_images ── Topic: 57
        │    │    │    │         └─printer_fonts_print_mouse_postscript
        │    │    │    │              ├─printer_fonts_print_font_deskjet
        │    │    │    │              │    ├─■──scanner_logitech_grayscale_ocr_scanman ── Topic: 108
        │    │    │    │              │    └─printer_fonts_print_font_deskjet
        │    │    │    │              │         ├─■──printer_print_deskjet_hp_ink ── Topic: 18
        │    │    │    │              │         └─■──fonts_font_truetype_tt_atm ── Topic: 49
        │    │    │    │              └─mouse_ghostscript_midi_driver_postscript
        │    │    │    │                   ├─ghostscript_midi_postscript_files_file
        │    │    │    │                   │    ├─■──ghostscript_postscript_pageview_ghostview_dsc ── Topic: 104
        │    │    │    │                   │    └─midi_sound_file_windows_driver
        │    │    │    │                   │         ├─■──location_mar_file_host_rwrr ── Topic: 83
        │    │    │    │                   │         └─■──midi_sound_driver_blaster_soundblaster ── Topic: 98
        │    │    │    │                   └─■──mouse_driver_mice_ball_problem ── Topic: 68
        │    │    │    └─game_team_games_25_season
        │    │    │         ├─1st_sale_condition_comics_hulk
        │    │    │         │    ├─sale_condition_offer_asking_cd
        │    │    │         │    │    ├─condition_stereo_amp_speakers_asking
        │    │    │         │    │    │    ├─■──miles_car_amfm_toyota_cassette ── Topic: 62
        │    │    │         │    │    │    └─■──amp_speakers_condition_stereo_audio ── Topic: 24
        │    │    │         │    │    └─games_sale_pom_cds_shipping
        │    │    │         │    │         ├─pom_cds_sale_shipping_cd
        │    │    │         │    │         │    ├─■──size_shipping_sale_condition_mattress ── Topic: 100
        │    │    │         │    │         │    └─■──pom_cds_cd_sale_picture ── Topic: 37
        │    │    │         │    │         └─■──games_game_snes_sega_genesis ── Topic: 40
        │    │    │         │    └─1st_hulk_comics_art_appears
        │    │    │         │         ├─1st_hulk_comics_art_appears
        │    │    │         │         │    ├─lens_tape_camera_backup_lenses
        │    │    │         │         │    │    ├─■──tape_backup_tapes_drive_4mm ── Topic: 107
        │    │    │         │         │    │    └─■──lens_camera_lenses_zoom_pouch ── Topic: 114
        │    │    │         │         │    └─1st_hulk_comics_art_appears
        │    │    │         │         │         ├─■──1st_hulk_comics_art_appears ── Topic: 105
        │    │    │         │         │         └─■──books_book_cover_trek_chemistry ── Topic: 125
        │    │    │         │         └─tickets_hotel_ticket_voucher_package
        │    │    │         │              ├─■──hotel_voucher_package_vacation_room ── Topic: 74
        │    │    │         │              └─■──tickets_ticket_june_airlines_july ── Topic: 84
        │    │    │         └─game_team_games_season_hockey
        │    │    │              ├─game_hockey_team_25_550
        │    │    │              │    ├─■──espn_pt_pts_game_la ── Topic: 17
        │    │    │              │    └─■──team_25_game_hockey_550 ── Topic: 2
        │    │    │              └─■──year_game_hit_baseball_players ── Topic: 0
        │    │    └─bike_car_greek_insurance_msg
        │    │         ├─car_bike_insurance_cars_engine
        │    │         │    ├─car_insurance_cars_radar_engine
        │    │         │    │    ├─insurance_health_private_care_canada
        │    │         │    │    │    ├─■──insurance_health_private_care_canada ── Topic: 99
        │    │         │    │    │    └─■──insurance_car_accident_rates_sue ── Topic: 82
        │    │         │    │    └─car_cars_radar_engine_detector
        │    │         │    │         ├─car_radar_cars_detector_engine
        │    │         │    │         │    ├─■──radar_detector_detectors_ka_alarm ── Topic: 39
        │    │         │    │         │    └─car_cars_mustang_ford_engine
        │    │         │    │         │         ├─■──clutch_shift_shifting_transmission_gear ── Topic: 88
        │    │         │    │         │         └─■──car_cars_mustang_ford_v8 ── Topic: 14
        │    │         │    │         └─oil_diesel_odometer_diesels_car
        │    │         │    │              ├─odometer_oil_sensor_car_drain
        │    │         │    │              │    ├─■──odometer_sensor_speedo_gauge_mileage ── Topic: 96
        │    │         │    │              │    └─■──oil_drain_car_leaks_taillights ── Topic: 102
        │    │         │    │              └─■──diesel_diesels_emissions_fuel_oil ── Topic: 79
        │    │         │    └─bike_riding_ride_bikes_motorcycle
        │    │         │         ├─bike_ride_riding_bikes_lane
        │    │         │         │    ├─■──bike_ride_riding_lane_car ── Topic: 11
        │    │         │         │    └─■──bike_bikes_miles_honda_motorcycle ── Topic: 19
        │    │         │         └─■──countersteering_bike_motorcycle_rear_shaft ── Topic: 46
        │    │         └─greek_msg_kuwait_greece_water
        │    │              ├─greek_msg_kuwait_greece_water
        │    │              │    ├─greek_msg_kuwait_greece_dog
        │    │              │    │    ├─greek_msg_kuwait_greece_dog
        │    │              │    │    │    ├─greek_kuwait_greece_turkish_greeks
        │    │              │    │    │    │    ├─■──greek_greece_turkish_greeks_cyprus ── Topic: 71
        │    │              │    │    │    │    └─■──kuwait_iraq_iran_gulf_arabia ── Topic: 76
        │    │              │    │    │    └─msg_dog_drugs_drug_food
        │    │              │    │    │         ├─dog_dogs_cooper_trial_weaver
        │    │              │    │    │         │    ├─■──clinton_bush_quayle_reagan_panicking ── Topic: 101
        │    │              │    │    │         │    └─dog_dogs_cooper_trial_weaver
        │    │              │    │    │         │         ├─■──cooper_trial_weaver_spence_witnesses ── Topic: 90
        │    │              │    │    │         │         └─■──dog_dogs_bike_trained_springer ── Topic: 67
        │    │              │    │    │         └─msg_drugs_drug_food_chinese
        │    │              │    │    │              ├─■──msg_food_chinese_foods_taste ── Topic: 30
        │    │              │    │    │              └─■──drugs_drug_marijuana_cocaine_alcohol ── Topic: 72
        │    │              │    │    └─water_theory_universe_science_larsons
        │    │              │    │         ├─water_nuclear_cooling_steam_dept
        │    │              │    │         │    ├─■──rocketry_rockets_engines_nuclear_plutonium ── Topic: 115
        │    │              │    │         │    └─water_cooling_steam_dept_plants
        │    │              │    │         │         ├─■──water_dept_phd_environmental_atmospheric ── Topic: 97
        │    │              │    │         │         └─■──cooling_water_steam_towers_plants ── Topic: 109
        │    │              │    │         └─theory_universe_larsons_larson_science
        │    │              │    │              ├─■──theory_universe_larsons_larson_science ── Topic: 54
        │    │              │    │              └─■──oort_cloud_grbs_gamma_burst ── Topic: 80
        │    │              │    └─helmet_kirlian_photography_lock_wax
        │    │              │         ├─helmet_kirlian_photography_leaf_mask
        │    │              │         │    ├─kirlian_photography_leaf_pictures_deleted
        │    │              │         │    │    ├─deleted_joke_stuff_maddi_nickname
        │    │              │         │    │    │    ├─■──joke_maddi_nickname_nicknames_frank ── Topic: 43
        │    │              │         │    │    │    └─■──deleted_stuff_bookstore_joke_motto ── Topic: 81
        │    │              │         │    │    └─■──kirlian_photography_leaf_pictures_aura ── Topic: 85
        │    │              │         │    └─helmet_mask_liner_foam_cb
        │    │              │         │         ├─■──helmet_liner_foam_cb_helmets ── Topic: 112
        │    │              │         │         └─■──mask_goalies_77_santore_tl ── Topic: 123
        │    │              │         └─lock_wax_paint_plastic_ear
        │    │              │              ├─■──lock_cable_locks_bike_600 ── Topic: 117
        │    │              │              └─wax_paint_ear_plastic_skin
        │    │              │                   ├─■──wax_paint_plastic_scratches_solvent ── Topic: 65
        │    │              │                   └─■──ear_wax_skin_greasy_acne ── Topic: 116
        │    │              └─m4_mp_14_mw_mo
        │    │                   ├─m4_mp_14_mw_mo
        │    │                   │    ├─■──m4_mp_14_mw_mo ── Topic: 111
        │    │                   │    └─■──test_ensign_nameless_deane_deanebinahccbrandeisedu ── Topic: 118
        │    │                   └─■──ites_cheek_hello_hi_ken ── Topic: 3
        │    └─space_medical_health_disease_cancer
        │         ├─medical_health_disease_cancer_patients
        │         │    ├─■──cancer_centers_center_medical_research ── Topic: 122
        │         │    └─health_medical_disease_patients_hiv
        │         │         ├─patients_medical_disease_candida_health
        │         │         │    ├─■──candida_yeast_infection_gonorrhea_infections ── Topic: 48
        │         │         │    └─patients_disease_cancer_medical_doctor
        │         │         │         ├─■──hiv_medical_cancer_patients_doctor ── Topic: 34
        │         │         │         └─■──pain_drug_patients_disease_diet ── Topic: 26
        │         │         └─■──health_newsgroup_tobacco_vote_votes ── Topic: 9
        │         └─space_launch_nasa_shuttle_orbit
        │              ├─space_moon_station_nasa_launch
        │              │    ├─■──sky_advertising_billboard_billboards_space ── Topic: 59
        │              │    └─■──space_station_moon_redesign_nasa ── Topic: 16
        │              └─space_mission_hst_launch_orbit
        │                   ├─space_launch_nasa_orbit_propulsion
        │                   │    ├─■──space_launch_nasa_propulsion_astronaut ── Topic: 47
        │                   │    └─■──orbit_km_jupiter_probe_earth ── Topic: 86
        │                   └─■──hst_mission_shuttle_orbit_arrays ── Topic: 60
        └─drive_file_key_windows_use
            ├─key_file_jpeg_encryption_image
            │    ├─key_encryption_clipper_chip_keys
            │    │    ├─■──key_clipper_encryption_chip_keys ── Topic: 1
            │    │    └─■──entry_file_ripem_entries_key ── Topic: 73
            │    └─jpeg_image_file_gif_images
            │         ├─motif_graphics_ftp_available_3d
            │         │    ├─motif_graphics_openwindows_ftp_available
            │         │    │    ├─■──openwindows_motif_xview_windows_mouse ── Topic: 20
            │         │    │    └─■──graphics_widget_ray_3d_available ── Topic: 95
            │         │    └─■──3d_machines_version_comments_contact ── Topic: 38
            │         └─jpeg_image_gif_images_format
            │              ├─■──gopher_ftp_files_stuffit_images ── Topic: 51
            │              └─■──jpeg_image_gif_format_images ── Topic: 13
            └─drive_db_card_scsi_windows
                ├─db_windows_dos_mov_os2
                │    ├─■──copy_protection_program_software_disk ── Topic: 64
                │    └─■──db_windows_dos_mov_os2 ── Topic: 8
                └─drive_card_scsi_drives_ide
                        ├─drive_scsi_drives_ide_disk
                        │    ├─■──drive_scsi_drives_ide_disk ── Topic: 6
                        │    └─■──meg_sale_ram_drive_shipping ── Topic: 12
                        └─card_modem_monitor_video_drivers
                            ├─■──card_monitor_video_drivers_vga ── Topic: 5
                            └─■──modem_port_serial_irq_com ── Topic: 10
  ```
</details>


## **Merge topics**

After seeing the potential hierarchy of your topic, you might want to merge specific
topics. For example, if topic 1 is
`1_space_launch_moon_nasa` and topic 2 is `2_spacecraft_solar_space_orbit` it might
make sense to merge those two topics as they are quite similar in meaning. In BERTopic,
you can use `.merge_topics` to manually select and merge those topics. Doing so will
update their topic representation which in turn updates the entire model:

```python
topics_to_merge = [1, 2]
topic_model.merge_topics(docs, topics_to_merge)
```

If you have several groups of topics you want to merge, create a list of lists instead:

```python
topics_to_merge = [[1, 2],
                   [3, 4]]
topic_model.merge_topics(docs, topics_to_merge)
```
