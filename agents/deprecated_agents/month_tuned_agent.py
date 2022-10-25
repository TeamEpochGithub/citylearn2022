import numpy as np
import os.path as osp
import csv
from traineval.evaluation import tuned_values as value_dir

def combined_policy(observation, action_space):
    values_path = osp.join(osp.dirname(value_dir.__file__), "optimal_values_month.csv")
    all_args = list(csv.DictReader(open(values_path), quoting=csv.QUOTE_ALL))

    # all_args = [{'carbon_1': '-0.9389486089896443', 'carbon_2': '-0.7381865387303848', 'carbon_3': '-0.7457949290604127', 'consumption_1': '0.6188683930709404', 'consumption_2': '0.40050025591631033', 'consumption_3': '0.7264708315063716', 'hour_1': '0.7910814224260633', 'hour_2': '-0.5968908792019267', 'hour_3': '0.2986355390196085', 'humidity_1': '0.4523732354909736', 'humidity_2': '0.3715846054675487', 'humidity_3': '0.21074177994613866', 'load_1': '0.7024298180892682', 'load_2': '-0.22536243921108895', 'load_3': '0.23597617956995992', 'price_1': '-0.23446347667784984', 'price_2': '-0.4689917290042636', 'price_3': '-0.6732494241557091', 'price_pred_1': '-0.037403029926924264', 'price_pred_2': '-0.4038714626939115', 'price_pred_3': '0.49369215108228437', 'solar_1': '-0.8777836470620023', 'solar_2': '0.49746459619441113', 'solar_3': '-0.40061583964740394', 'solar_diffused_1': '0.6144679318020221', 'solar_diffused_2': '-0.6605591005622575', 'solar_diffused_3': '-0.4696085049087386', 'solar_direct_1': '0.4439512420328433', 'solar_direct_2': '0.7314011371765171', 'solar_direct_3': '0.6338869818641913', 'storage_1': '0.27986639412930975', 'storage_2': '0.5573990399728159', 'storage_3': '0.11198351360207902', 'temp_1': '0.00829379555217194', 'temp_2': '-0.38884818615000466', 'temp_3': '-0.15073767899763957', 'month': '1'}, {'carbon_1': '-0.69169928825966', 'carbon_2': '-0.11271924033022815', 'carbon_3': '-0.11633795007800657', 'consumption_1': '-0.07743040944817964', 'consumption_2': '0.017690548802479306', 'consumption_3': '0.604493667556957', 'hour_1': '-0.2738319147870181', 'hour_2': '-0.6773195786199344', 'hour_3': '-0.37142653972480577', 'humidity_1': '0.5093556367139431', 'humidity_2': '0.904386093290567', 'humidity_3': '0.8732268552028796', 'load_1': '0.2671567076532203', 'load_2': '0.689321778715098', 'load_3': '0.9033308626147765', 'price_1': '0.8834834232117701', 'price_2': '0.4446882461612858', 'price_3': '0.3013455838524901', 'price_pred_1': '-0.9323132601852219', 'price_pred_2': '0.04331367746599277', 'price_pred_3': '-0.412899785753015', 'solar_1': '0.9994918142513605', 'solar_2': '0.22822725199523444', 'solar_3': '-0.7738062604600776', 'solar_diffused_1': '0.6836679824883191', 'solar_diffused_2': '0.20148013985116753', 'solar_diffused_3': '0.2815092128840574', 'solar_direct_1': '-0.3733582555290944', 'solar_direct_2': '-0.46472513959866213', 'solar_direct_3': '-0.4229814220870744', 'storage_1': '-0.3485414778643792', 'storage_2': '-0.02382143771726314', 'storage_3': '-0.7166061866316111', 'temp_1': '0.5398395970830168', 'temp_2': '0.738138106985538', 'temp_3': '0.6702878106396706', 'month': '2'}, {'carbon_1': '0.4167433664389228', 'carbon_2': '0.5393427002835766', 'carbon_3': '0.5270113795040391', 'consumption_1': '-0.7102410215346581', 'consumption_2': '-0.4554718416926599', 'consumption_3': '-0.7298535714632005', 'hour_1': '0.7458752233198982', 'hour_2': '-0.14439489267954891', 'hour_3': '-0.05676827193916343', 'humidity_1': '0.10594722154371625', 'humidity_2': '0.3225157301583199', 'humidity_3': '0.28067887090846527', 'load_1': '0.7414271178166243', 'load_2': '-0.7046232551410012', 'load_3': '-0.51683262057838', 'price_1': '0.6133176822496005', 'price_2': '0.8769242855462703', 'price_3': '0.3240099896255556', 'price_pred_1': '-0.2634279561679139', 'price_pred_2': '0.7072174278247756', 'price_pred_3': '0.7040700027060591', 'solar_1': '0.22406115220572986', 'solar_2': '0.4300151352623075', 'solar_3': '-0.5917686744590912', 'solar_diffused_1': '-0.7490616003188114', 'solar_diffused_2': '-0.328503718017354', 'solar_diffused_3': '0.1231679443058378', 'solar_direct_1': '0.5144706444921838', 'solar_direct_2': '-0.7970834218250168', 'solar_direct_3': '-0.7896523982470091', 'storage_1': '-0.07137983593655309', 'storage_2': '-0.11096610769891214', 'storage_3': '-0.4037618270445653', 'temp_1': '-0.8302429804410815', 'temp_2': '-0.6322046310726172', 'temp_3': '-0.5176950745714048', 'month': '3'}, {'carbon_1': '0.6897851879405427', 'carbon_2': '0.7572051976635473', 'carbon_3': '0.5149741480474999', 'consumption_1': '0.10751525155073993', 'consumption_2': '0.1307407913665167', 'consumption_3': '0.8511531718503704', 'hour_1': '0.8451574763996017', 'hour_2': '-0.5347395337322621', 'hour_3': '-0.4949545888142445', 'humidity_1': '0.5749289434262101', 'humidity_2': '0.21631390240884882', 'humidity_3': '0.38145101747337606', 'load_1': '0.47493690462113247', 'load_2': '0.765030722401966', 'load_3': '-0.22735288676650175', 'price_1': '-0.5231124565291347', 'price_2': '-0.739153083787681', 'price_3': '-0.7554655690653617', 'price_pred_1': '0.10089007756332971', 'price_pred_2': '-0.1680271187262628', 'price_pred_3': '0.7968567709679268', 'solar_1': '-0.3775624986737824', 'solar_2': '-0.26794682386223223', 'solar_3': '-0.8145683571177745', 'solar_diffused_1': '0.8137355576533818', 'solar_diffused_2': '-0.8155072801448257', 'solar_diffused_3': '0.34142828735199593', 'solar_direct_1': '0.9995713392808705', 'solar_direct_2': '0.7089668176986423', 'solar_direct_3': '0.19867530554795826', 'storage_1': '-0.54663328370222', 'storage_2': '-0.21695065557301416', 'storage_3': '-0.771304449848896', 'temp_1': '-0.4670740146596291', 'temp_2': '-0.9281905918449287', 'temp_3': '-0.4622989943438447', 'month': '4'}, {'carbon_1': '-0.3115742516010181', 'carbon_2': '-0.12454320052565016', 'carbon_3': '-0.24316960455017606', 'consumption_1': '0.0631991107600059', 'consumption_2': '-0.6313462385274422', 'consumption_3': '0.6560609458247748', 'hour_1': '0.9624053120042344', 'hour_2': '-0.8850261220779929', 'hour_3': '-0.9977598268715301', 'humidity_1': '-0.39075628029839476', 'humidity_2': '-0.6877509625578765', 'humidity_3': '-0.5863432944004862', 'load_1': '0.07902429515052554', 'load_2': '0.10321490507795505', 'load_3': '-0.6871314608790596', 'price_1': '0.5078113194038808', 'price_2': '0.5380331929188054', 'price_3': '0.08694363330475344', 'price_pred_1': '-0.5125285727549255', 'price_pred_2': '0.7375159204756225', 'price_pred_3': '-0.40715800590419693', 'solar_1': '-0.6623776737342961', 'solar_2': '-0.005951481396545043', 'solar_3': '-0.6304339137325949', 'solar_diffused_1': '0.3013792491554987', 'solar_diffused_2': '-0.23824579562326967', 'solar_diffused_3': '0.8387588092928155', 'solar_direct_1': '0.2991351169018193', 'solar_direct_2': '0.33234178210382676', 'solar_direct_3': '-0.11107985981569288', 'storage_1': '0.349553587096783', 'storage_2': '0.9692358687057767', 'storage_3': '0.36544169935143167', 'temp_1': '0.10099947220609083', 'temp_2': '0.030332425719161976', 'temp_3': '0.623915289907165', 'month': '5'}, {'carbon_1': '0.9639565824020456', 'carbon_2': '0.9461559655306685', 'carbon_3': '0.800901396960535', 'consumption_1': '-0.9597290792260649', 'consumption_2': '0.04840820992000908', 'consumption_3': '-0.36878165007294594', 'hour_1': '0.9662374627984466', 'hour_2': '-0.15055587144311164', 'hour_3': '-0.8394447544611326', 'humidity_1': '0.7205811376780566', 'humidity_2': '0.7823801017466376', 'humidity_3': '0.20011348530275835', 'load_1': '0.3278937100552304', 'load_2': '-0.881704812134706', 'load_3': '-0.9182815181725309', 'price_1': '0.22090115216837042', 'price_2': '0.23305070303485312', 'price_3': '-0.6286717385017621', 'price_pred_1': '-0.99835180940402', 'price_pred_2': '0.24796454496816403', 'price_pred_3': '0.5457716241112712', 'solar_1': '0.5503292837223019', 'solar_2': '-0.7129010650529913', 'solar_3': '-0.9988378452218905', 'solar_diffused_1': '0.6437715216098104', 'solar_diffused_2': '0.5445240727913666', 'solar_diffused_3': '-0.0028901449746136926', 'solar_direct_1': '0.34872724478541345', 'solar_direct_2': '-0.9346677161074213', 'solar_direct_3': '-0.9970427219144476', 'storage_1': '0.28921091299087254', 'storage_2': '0.914663656236702', 'storage_3': '0.7749162884567377', 'temp_1': '-0.5342756596026724', 'temp_2': '-0.5888258240094487', 'temp_3': '0.37768981510316885', 'month': '6'}, {'carbon_1': '-0.11052793652379699', 'carbon_2': '0.0037419554367454644', 'carbon_3': '0.014895501388825713', 'consumption_1': '-0.9741969196487505', 'consumption_2': '-0.8527484263071475', 'consumption_3': '-0.14421488727290807', 'hour_1': '0.6716145709550543', 'hour_2': '-0.8022457495021865', 'hour_3': '0.350152287285', 'humidity_1': '-0.722913066273625', 'humidity_2': '-0.7824234567368885', 'humidity_3': '-0.7095929686608843', 'load_1': '0.668832527524104', 'load_2': '0.9133021690458788', 'load_3': '0.1632584552867029', 'price_1': '0.923066522512214', 'price_2': '0.8904740473329027', 'price_3': '-0.7072517244214698', 'price_pred_1': '0.3445637514297849', 'price_pred_2': '-0.9988069655932323', 'price_pred_3': '-0.5969610539854227', 'solar_1': '-0.2662486448344142', 'solar_2': '0.8674391574221594', 'solar_3': '-0.7830047138708895', 'solar_diffused_1': '-0.9354308684515501', 'solar_diffused_2': '0.4266021651555804', 'solar_diffused_3': '0.6530506636869564', 'solar_direct_1': '-0.9318986356192929', 'solar_direct_2': '-0.2810654080092742', 'solar_direct_3': '-0.2482212113326867', 'storage_1': '0.25473000907720045', 'storage_2': '-0.2878335657444103', 'storage_3': '-0.046311621852197046', 'temp_1': '-0.6907416846753671', 'temp_2': '-0.06082076508695468', 'temp_3': '0.46549067279197953', 'month': '7'}, {'carbon_1': '0.26311275116503363', 'carbon_2': '0.03977183517202593', 'carbon_3': '0.14367486145181607', 'consumption_1': '0.5157740302503372', 'consumption_2': '-0.420089407620833', 'consumption_3': '0.8894960463608038', 'hour_1': '0.9599354313202786', 'hour_2': '-0.8705759740185561', 'hour_3': '-0.07839348211838792', 'humidity_1': '-0.8669381098627731', 'humidity_2': '-0.8784582658115176', 'humidity_3': '-0.9261734373036941', 'load_1': '0.08194631468336928', 'load_2': '-0.4988399730995717', 'load_3': '-0.5200741269474642', 'price_1': '0.9226177038216918', 'price_2': '0.867910711434908', 'price_3': '0.24943076197281777', 'price_pred_1': '0.07202660944623777', 'price_pred_2': '-0.8115758178556243', 'price_pred_3': '-0.26613539795630475', 'solar_1': '0.6840331962622921', 'solar_2': '-0.41659228522826947', 'solar_3': '-0.8881849649746576', 'solar_diffused_1': '0.8327834615956653', 'solar_diffused_2': '-0.14327814081583354', 'solar_diffused_3': '0.6026669526468583', 'solar_direct_1': '0.9266215062387753', 'solar_direct_2': '0.9972729363728106', 'solar_direct_3': '0.42334935380497135', 'storage_1': '0.8246327989816884', 'storage_2': '0.6374896633654731', 'storage_3': '0.8850960607005893', 'temp_1': '0.8579597617807655', 'temp_2': '-0.8501974365986247', 'temp_3': '-0.8663636322408459', 'month': '8'}, {'carbon_1': '-0.18822668188245523', 'carbon_2': '-0.6806260699024995', 'carbon_3': '-0.53178380902978', 'consumption_1': '0.0435483604766969', 'consumption_2': '-0.8882726585432615', 'consumption_3': '0.9619239209915207', 'hour_1': '0.9995618693547019', 'hour_2': '-0.6113754795456933', 'hour_3': '0.014467404180798199', 'humidity_1': '-0.9547718010341474', 'humidity_2': '-0.820685816087943', 'humidity_3': '-0.7363559949708238', 'load_1': '-0.2044458819367147', 'load_2': '0.4689383334817897', 'load_3': '-0.8178716691582092', 'price_1': '0.8948187139315384', 'price_2': '0.5755114051585575', 'price_3': '-0.0611325895634289', 'price_pred_1': '-0.11910998472889332', 'price_pred_2': '-0.06788142517665739', 'price_pred_3': '0.3458887647903278', 'solar_1': '0.12228759772683218', 'solar_2': '-0.06282303687845031', 'solar_3': '-0.5707127602616783', 'solar_diffused_1': '0.9987265733382737', 'solar_diffused_2': '-0.4733401950566647', 'solar_diffused_3': '-0.07854686674887527', 'solar_direct_1': '0.3882580853623334', 'solar_direct_2': '0.419850438541124', 'solar_direct_3': '0.4497085971139151', 'storage_1': '-0.3596528949627834', 'storage_2': '0.20542843039522576', 'storage_3': '-0.4355540832940187', 'temp_1': '-0.02439399229689998', 'temp_2': '-0.3061830820218696', 'temp_3': '0.21418256188950077', 'month': '9'}, {'carbon_1': '0.008856910087628067', 'carbon_2': '-0.0360312846899849', 'carbon_3': '0.29127726309715835', 'consumption_1': '-0.604738750797191', 'consumption_2': '-0.41071460790684267', 'consumption_3': '0.6248024551973917', 'hour_1': '0.9569605353341563', 'hour_2': '-0.7366718755474528', 'hour_3': '0.5864907004463379', 'humidity_1': '-0.24439435876527787', 'humidity_2': '0.2741850090805137', 'humidity_3': '-0.06080745540826156', 'load_1': '0.2210914533214607', 'load_2': '-0.9757361091357114', 'load_3': '-0.819627662207431', 'price_1': '-0.5061296963036661', 'price_2': '-0.28804798166872736', 'price_3': '-0.9352603116435603', 'price_pred_1': '0.46357429110366133', 'price_pred_2': '0.6652092867872047', 'price_pred_3': '0.8353658928734529', 'solar_1': '0.33177785437712515', 'solar_2': '0.10652781305479327', 'solar_3': '-0.7759485504446687', 'solar_diffused_1': '0.5696477187040202', 'solar_diffused_2': '0.6130361327558279', 'solar_diffused_3': '0.8714823084702042', 'solar_direct_1': '-0.2766473577829288', 'solar_direct_2': '-0.9943013190745137', 'solar_direct_3': '-0.8232686217253373', 'storage_1': '0.7872487668993455', 'storage_2': '0.6058304116674207', 'storage_3': '0.504505826413564', 'temp_1': '-0.6090806088871114', 'temp_2': '-0.06830273383818505', 'temp_3': '0.10446422857449722', 'month': '10'}, {'carbon_1': '-0.3870907048825099', 'carbon_2': '-0.5518989310548287', 'carbon_3': '-0.2599879293477891', 'consumption_1': '-0.56051304678903', 'consumption_2': '-0.9250339639989252', 'consumption_3': '0.6299666528912559', 'hour_1': '0.8618702127545226', 'hour_2': '-0.894044342149372', 'hour_3': '0.4023847242181899', 'humidity_1': '0.5018911539264053', 'humidity_2': '0.4116576299962718', 'humidity_3': '0.0002485655437667613', 'load_1': '-0.7475081563409617', 'load_2': '-0.17274464719038513', 'load_3': '-0.869230932814909', 'price_1': '-0.2354077828474795', 'price_2': '0.8157058145498178', 'price_3': '-0.8535383455553698', 'price_pred_1': '0.6653718570222908', 'price_pred_2': '-0.5139442829157257', 'price_pred_3': '0.9577890085037896', 'solar_1': '0.6004620946385837', 'solar_2': '0.11859341393013206', 'solar_3': '-0.48700562883623466', 'solar_diffused_1': '-0.638366648159312', 'solar_diffused_2': '0.9138592329042066', 'solar_diffused_3': '0.7528227441744517', 'solar_direct_1': '-0.033393395615460245', 'solar_direct_2': '-0.21102332755589265', 'solar_direct_3': '-0.21886321389357793', 'storage_1': '-0.7053906209666451', 'storage_2': '-0.20301183729412936', 'storage_3': '-0.9322359079787365', 'temp_1': '-0.11981103818245877', 'temp_2': '-0.06060499082290945', 'temp_3': '-0.5675388655469826', 'month': '11'}, {'carbon_1': '0.08745070335241029', 'carbon_2': '0.2022126457187506', 'carbon_3': '-0.25452888856732353', 'consumption_1': '-0.02182358945686231', 'consumption_2': '-0.04976925636631206', 'consumption_3': '0.8316548225526962', 'hour_1': '0.6469087722526851', 'hour_2': '-0.9285479902045037', 'hour_3': '-0.2724819160996586', 'humidity_1': '0.010836956411927733', 'humidity_2': '-0.4174559293594271', 'humidity_3': '0.1700699474606156', 'load_1': '0.8210279597962643', 'load_2': '0.6407089791597619', 'load_3': '0.9260181483186072', 'price_1': '0.7620726118516842', 'price_2': '-0.9613080387092441', 'price_3': '0.27668527884135863', 'price_pred_1': '-0.7467020156543889', 'price_pred_2': '0.23165884319894026', 'price_pred_3': '-0.23741675271026208', 'solar_1': '-0.31331794517499645', 'solar_2': '0.5029283028953392', 'solar_3': '-0.1104645454511156', 'solar_diffused_1': '-0.08175246724614937', 'solar_diffused_2': '-0.05224003841917642', 'solar_diffused_3': '-0.45316195236551304', 'solar_direct_1': '-0.19663271276254027', 'solar_direct_2': '-0.04869162197183367', 'solar_direct_3': '0.11495457147794849', 'storage_1': '-0.8380139480394286', 'storage_2': '-0.7476425277933113', 'storage_3': '-0.800671740759251', 'temp_1': '0.06397380789983956', 'temp_2': '-0.28624570851519404', 'temp_3': '-0.5637920637421617', 'month': '12'}]

    month = observation[0]
    args = None
    for arg_dict in all_args:
        if int(arg_dict["month"]) == int(month):
            args = arg_dict
            break
    # args = all_args[int(month) - 1]
    for k, v in args.items():
        args[k] = float(v)

    day_type = observation[1]
    hour = observation[2]
    outdoor_dry_bulb_temperature = observation[3]
    outdoor_dry_bulb_temperature_predicted_6h = observation[4]
    outdoor_dry_bulb_temperature_predicted_12h = observation[5]
    outdoor_dry_bulb_temperature_predicted_24h = observation[6]
    outdoor_relative_humidity = observation[7]
    outdoor_relative_humidity_predicted_6h = observation[8]
    outdoor_relative_humidity_predicted_12h = observation[9]
    outdoor_relative_humidity_predicted_24h = observation[10]
    diffuse_solar_irradiance = observation[11]
    diffuse_solar_irradiance_predicted_6h = observation[12]
    diffuse_solar_irradiance_predicted_12h = observation[13]
    diffuse_solar_irradiance_predicted_24h = observation[14]
    direct_solar_irradiance = observation[15]
    direct_solar_irradiance_predicted_6h = observation[16]
    direct_solar_irradiance_predicted_12h = observation[17]
    direct_solar_irradiance_predicted_24h = observation[18]
    carbon_intensity = observation[19]
    non_shiftable_load = observation[20]
    solar_generation = observation[21]
    electrical_storage_soc = observation[22]
    net_electricity_consumption = observation[23]
    electricity_pricing = observation[24]
    electricity_pricing_predicted_6h = observation[25]
    electricity_pricing_predicted_12h = observation[26]
    electricity_pricing_predicted_24h = observation[27]

    ### PRICE
    pricing_action = 1
    if 0 < electricity_pricing <= 0.21:
        pricing_action *= args["price_1"]
    elif 0.21 < electricity_pricing <= 0.22:
        pricing_action *= args["price_2"]
    else:
        pricing_action *= args["price_3"]

    pricing_pred_action = 1
    if 0 < electricity_pricing_predicted_6h <= 0.21:
        pricing_pred_action *= args["price_pred_1"]
    elif 0.21 < electricity_pricing_predicted_6h <= 0.22:
        pricing_pred_action *= args["price_pred_2"]
    else:
        pricing_pred_action *= args["price_pred_3"]

    ### EMISSION
    carbon_action = 1
    if 0 < carbon_intensity <= 0.139231:
        carbon_action *= args["carbon_1"]
    elif 0.139231 < carbon_intensity <= 0.169461:
        carbon_action *= args["carbon_2"]
    else:
        carbon_action *= args["carbon_3"]

    generation_action = 1
    if 0 < solar_generation <= 0:
        generation_action *= args["solar_1"]
    elif 0 < solar_generation <= 163.14452:
        generation_action *= args["solar_2"]
    else:
        generation_action *= args["solar_3"]

    diffuse_action = 1
    if 0 < diffuse_solar_irradiance <= 0:
        diffuse_action *= args["solar_diffused_1"]
    elif 0 < diffuse_solar_irradiance <= 216:
        diffuse_action *= args["solar_diffused_2"]
    else:
        diffuse_action *= args["solar_diffused_3"]

    direct_action = 1
    if 0 < direct_solar_irradiance <= 0:
        direct_action *= args["solar_direct_1"]
    elif 0 < direct_solar_irradiance <= 141:
        direct_action *= args["solar_direct_2"]
    else:
        direct_action *= args["solar_direct_3"]

    ### GRID
    hour_action = 1
    if 6 < hour <= 14:
        hour_action *= args["hour_1"]
    elif 14 < hour <= 23:
        hour_action *= args["hour_2"]
    else:
        hour_action *= args["hour_3"]

    storage_action = 1
    if 0 < electrical_storage_soc <= 0.33:
        storage_action *= args["storage_1"]
    elif 0.33 < electrical_storage_soc <= 0.66:
        storage_action *= args["storage_2"]
    else:
        storage_action *= args["storage_3"]

    consumption_action = 1
    if 0 < net_electricity_consumption <= 0.6:
        consumption_action *= args["consumption_1"]
    elif 0.6 < net_electricity_consumption <= 1.2:
        consumption_action *= args["consumption_2"]
    else:
        consumption_action *= args["consumption_3"]

    load_action = 1
    if 0 < non_shiftable_load <= 0.726493:
        load_action *= args["load_1"]
    elif 0.726493 < non_shiftable_load <= 1.185376:
        load_action *= args["load_2"]
    else:
        load_action *= args["load_3"]

    temp_action = 1
    if 0 < outdoor_dry_bulb_temperature <= 15.6:
        temp_action *= args["temp_1"]
    elif 15.6 < outdoor_dry_bulb_temperature <= 18.3:
        temp_action *= args["temp_2"]
    else:
        temp_action *= args["temp_3"]

    humidity_action = 1
    if 0 < outdoor_relative_humidity <= 69:
        humidity_action *= args["humidity_1"]
    elif 69 < outdoor_relative_humidity <= 81:
        humidity_action *= args["humidity_2"]
    else:
        humidity_action *= args["humidity_3"]

    price_action = np.average([pricing_action, pricing_pred_action])
    emission_action = np.average([carbon_action, generation_action, diffuse_action, direct_action])
    grid_action = np.average(
        [hour_action, storage_action, consumption_action, load_action, temp_action, humidity_action])

    action_average = (price_action + emission_action + grid_action) / 3

    action = np.array([action_average], dtype=action_space.dtype)

    return action


class MonthTunedAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return combined_policy(observation, self.action_space[agent_id])
