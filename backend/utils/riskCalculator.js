export function calculateRisk(
  weatherData,
  cropType,
  cropSensitivity,
  extras = {}
) {
  const { temperature, rainfall } = weatherData;
  const { soilType = "medium", irrigation = false } = extras;

  let limits;

  if (Array.isArray(cropSensitivity)) {
    const match = cropSensitivity.find(
      (c) => c.crop.trim().toLowerCase() === cropType.trim().toLowerCase()
    );

    if (match) {
      limits = {
        minRainfall: Number(match.minRainfall),
        maxRainfall: Number(match.maxRainfall),
        maxComfortTemp: Number(match.maxComfortTemp),
      };
    }
  }

  if (!limits) {
    const maize = cropSensitivity.find(
      (c) => c.crop.trim().toLowerCase() === "maize"
    );

    if (maize) {
      console.warn(
        ` Crop data not found for "${cropType}", using maize defaults.`
      );
      limits = {
        minRainfall: Number(maize.minRainfall),
        maxRainfall: Number(maize.maxRainfall),
        maxComfortTemp: Number(maize.maxComfortTemp),
      };
    } else {
      console.warn(
        `⚠️ Crop data missing for "${cropType}" and "maize". Using generic defaults.`
      );
      limits = {
        minRainfall: 50,
        maxRainfall: 200,
        maxComfortTemp: 30,
      };
    }
  }

  // Drought Risk 
  let droughtRisk = 1;
  const rainfallDeficit = limits.minRainfall - rainfall;
  if (rainfallDeficit > 0) droughtRisk = Math.min(5, 2 + rainfallDeficit / 20);
  if (temperature > limits.maxComfortTemp + 2) droughtRisk += 1;
  if (soilType === "sandy") droughtRisk += 1;
  if (irrigation) droughtRisk -= 1;
  droughtRisk = clamp(droughtRisk);

  //  Flood Risk 
  let floodRisk = 1;
  const rainfallExcess = rainfall - limits.maxRainfall;
  if (rainfallExcess > 0) floodRisk = Math.min(5, 2 + rainfallExcess / 30);
  if (soilType === "clay") floodRisk += 1;
  floodRisk = clamp(floodRisk);

  // Heat Risk 
  let heatRisk = 1;
  if (temperature > limits.maxComfortTemp) {
    heatRisk += (temperature - limits.maxComfortTemp) / 2;
  }
  heatRisk = clamp(heatRisk);

  return { droughtRisk, floodRisk, heatRisk };
}

export function generateRecommendations(riskScores, crop) {
  let rec = [];
  const { droughtRisk, floodRisk, heatRisk } = riskScores;

  if (droughtRisk >= 4) {
    rec.push(
      `Severe drought risk → consider drought-tolerant ${crop},we suggest mulching and drip irrigation.In regions facing a
       severe risk of drought, it is important to adopt strategies that help crops survive and thrive under limited water conditions. 
       Planting drought-tolerant varieties can ensure better yields despite water scarcity, while applying mulch around plants helps
        retain soil moisture, regulate temperature, and suppress weeds. In addition, using efficient irrigation methods such as drip 
        irrigation delivers water directly to the roots, reducing wastage and maximizing the benefits of available water. By combining 
        drought-resistant crops with mulching and targeted irrigation, farmers can significantly reduce the negative impacts of prolonged
         dry periods and maintain healthy, productive fields.`
    );
  } else if (droughtRisk >= 3) {
    rec.push(
      `Moderate drought risk → Apply mulch & schedule irrigation efficiently.In areas experiencing a moderate risk of drought, 
      it is advisable to use crops that are somewhat drought-tolerant and can withstand short periods of water stress. Practices 
      such as light mulching can help conserve soil moisture, while scheduling irrigation efficiently ensures that crops receive 
      adequate water without wastage. Farmers can also consider soil management techniques, such as adding organic matter, to 
      improve water retention and maintain soil health. By combining these approaches, it is possible to reduce the effects of 
      moderate drought and sustain crop growth with minimal risk.`
    );
  }

  if (floodRisk >= 4) {
    rec.push(
      `Severe flood risk → we suggest Improve drainage, raised beds & avoid water-logging.In regions facing a severe risk of flooding, 
      it is crucial to implement measures that protect crops and soil from waterlogging and erosion. Planting flood-tolerant crop
       varieties can help ensure survival and maintain yields during prolonged inundation, while constructing raised beds or ridges 
       allows water to drain more effectively from the root zone. In addition, proper field drainage systems, such as ditches or tile 
       drains, can prevent standing water and reduce the risk of crop damage. By combining flood-tolerant crops with strategic land 
       management and drainage practices, farmers can minimize losses and maintain productive fields even under severe flood conditions.`
    );
  } else if (floodRisk >= 3) {
    rec.push(`Moderate flood risk → we suggest Ensure proper drainage channels. In areas with a moderate risk of flooding, it is
       important to take preventive measures to protect crops and maintain soil health. Planting moderately flood-tolerant crop 
       varieties can reduce the impact of temporary waterlogging, while creating shallow drainage channels or slightly raised beds
        can help excess water move away from the fields. Additionally, careful monitoring of weather and water levels allows timely 
        interventions, such as adjusting planting schedules or applying protective measures. By combining these strategies, farmers
         can reduce the effects of moderate flooding and sustain crop growth with minimal damage.`);
  }

  if (heatRisk >= 4) {
    rec.push(
      `High heat stress → use heat-tolerant ${crop} seeds, we suggest to provide shade nets & irrigate during peak heat. 
      In regions experiencing high heat stress, it is essential to adopt strategies that protect crops from excessive temperatures 
      and prevent yield losses. Planting heat-tolerant crop varieties can help ensure survival and maintain productivity under 
      extreme heat conditions. Additionally, practices such as mulching and maintaining adequate soil moisture can reduce soil 
      temperature and protect plant roots, while timely irrigation—preferably during cooler parts of the day—can minimize water stress. 
      Providing temporary shade for sensitive crops and adjusting planting schedules to avoid peak heat periods can further improve resilience.
       By combining heat-tolerant crops with soil management and irrigation strategies, farmers can mitigate the negative impacts
        of high temperatures on their fields.`
    );
  } else if (heatRisk >= 3) {
    rec.push(
      `Moderate heat stress → we suggest to Apply irrigation during evenings; avoid midday stress. In areas experiencing moderate heat stress, 
      it is important to take measures that help crops cope with elevated temperatures without causing significant damage. 
      Planting crop varieties that are somewhat heat-tolerant can improve resilience, while practices such as mulching and 
      maintaining adequate soil moisture can help regulate soil temperature and reduce water stress. Additionally, adjusting 
      irrigation schedules to provide water during the warmer parts of the day and monitoring crop conditions can prevent heat-related stress.
       By combining these approaches, farmers can sustain healthy crop growth and minimize the effects of moderate heat stress.`
    );
  }

  if (rec.length === 0) {
    rec.push(
      `Risk levels normal → Follow standard agronomy practices for ${crop}. we suggest you to Standard agricultural practices, 
      including regular irrigation, routine soil management, and timely planting, are usually sufficient to maintain healthy
       crop growth and ensure good yields. Farmers can continue monitoring weather and soil conditions to stay prepared for
        any unexpected changes, but intensive protective measures are typically not required under normal risk conditions.`
    );
  }

  return rec;
}

function clamp(num, min = 1, max = 5) {
  return Math.max(min, Math.min(num, max));
}
