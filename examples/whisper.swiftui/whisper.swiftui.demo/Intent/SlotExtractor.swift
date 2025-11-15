import Foundation
import os.log

struct SlotExtractionResult {
    let slots: [String: Any]
    let confidence: Float
}

class SlotExtractor {
    
    // MARK: - Constants
    
    private static let logger = Logger(subsystem: "com.whispercpp.demo", category: "SlotExtractor")
    
    // MARK: - Slot Templates
    
    private let intentSlotTemplates: [String: [String]] = [
        "QueryPoint": ["metric"],  // time_ref, unit, and identifier can have defaults
        "SetGoal": ["metric", "target"],  // unit can be inferred
        "SetThreshold": ["metric", "threshold", "type"],  // unit can be inferred
        "TimerStopwatch": ["tool", "action"],  // value only required for "set timer X min"
        "ToggleFeature": ["feature", "state"],
        "LogEvent": ["event_type"],  // value, unit, time_ref can have defaults
        "StartActivity": ["activity_type"],
        "StopActivity": ["activity_type"],
        "OpenApp": ["app"],  // action and target can have defaults
        "PhoneAction": ["contact"], // no need of action
        "MediaAction": ["action"],  // target can be optional for some actions
        "WeatherQuery": ["location"],  // attribute can default to general weather
        "QueryTrend": ["metric"],  // period and unit can have defaults
        "IrrelevantInput": []  // No required slots for irrelevant input
    ]
    
    func getRequiredSlots(_ intent: String) -> [String] {
        return intentSlotTemplates[intent] ?? []
    }
    
    // Special method to check if slots are adequate for the specific action within an intent
    func areRequiredSlotsSatisfied(_ intent: String, slots: [String: Any], text: String) -> Bool {
        let baseRequiredSlots = getRequiredSlots(intent)
        
        // Check if all base required slots are present
        if !baseRequiredSlots.allSatisfy({ slots.keys.contains($0) }) {
            return false
        }
        
        // Special case handling for TimerStopwatch - value required only for "set timer"
        if intent == "TimerStopwatch" {
            if let action = slots["action"] as? String,
               let tool = slots["tool"] as? String {
                // If setting a timer, value is required
                if (action == "set" || action == "start") && tool == "timer" {
                    if slots["value"] == nil {
                        return false
                    }
                }
            }
        }
        
        // Special case for SetGoal and SetThreshold - target/threshold must be meaningful
        if intent == "SetGoal" && slots["target"] == nil {
            return false
        }
        
        if intent == "SetThreshold" && slots["threshold"] == nil {
            return false
        }
        
        return true
    }
    
    // MARK: - Synonym Mappings based on Kotlin implementation
    
    private let metricSynonyms: [String: [String]] = [
        "steps": [
            "steps", "step", "walk", "walked", "walking", "footsteps", "pace", "stride",
            "tread", "footfall", "gait", "paces", "stroll", "strolling", "strolled",
            "amble", "ambling", "saunter", "march", "marching", "trudge", "hike",
            "hiking", "trek", "wander", "movement", "activity", "moves"
        ],
        "distance": [
            "distance", "walked", "walk", "miles", "kilometers", "km", "far", "meter",
            "metres", "meters", "travelled", "traveled", "covered", "journey", "range",
            "length", "span", "route", "path", "mileage", "odometer", "how far",
            "feet", "yards"
        ],
        "calories": [
            "calories", "calorie", "kcal", "energy", "burned", "burn", "burning",
            "burnt", "expended", "consumed", "expenditure", "kilojoules", "kj",
            "nutrition", "intake", "food energy", "metabolic", "metabolism",
            "fat burn", "fat burning", "cal"
        ],
        "heart rate": [
            "heart rate", "heartrate", "hr", "pulse", "bpm", "heart beat", "heartbeat",
            "cardiac", "beats per minute", "resting heart rate", "rhr", "heart rhythm",
            "cardiovascular", "cardio", "heart health", "ticker", "heart pulse",
            "pulse rate", "heart monitor", "heart"
        ],
        "sleep": [
            "sleep", "slept", "sleeping", "rest", "rested", "resting", "nap",
            "napping", "napped", "slumber", "snooze", "snoozed", "doze", "dozed",
            "asleep", "bedtime", "night sleep", "shuteye", "zzz", "sleep time",
            "sleep duration", "hours slept", "sleep hours"
        ],
        "sleep score": [
            "sleep score", "sleep quality", "sleep rating", "sleep performance",
            "sleep analysis", "sleep grade", "sleep rank", "sleep level",
            "sleep efficiency", "sleep assessment", "how well slept", "sleep health",
            "sleep metric", "sleep stats", "sleep report", "sleep evaluation"
        ],
        "spo2": [
            "spo2", "oxygen", "blood oxygen", "o2", "saturation", "oxygen saturation",
            "oxygen level", "oxygen levels", "o2 sat", "blood o2", "oxygen sat",
            "pulse ox", "pulse oximetry", "oximeter", "oxygen reading", "o2 level",
            "respiratory", "breathing", "blood oxygen level"
        ],
        "weight": [
            "weight", "weigh", "kg", "pounds", "lbs", "kilogram", "kilograms", "lb",
            "body weight", "mass", "scale", "weighed", "weighing", "bmi",
            "body mass", "how much weigh", "weight reading", "weighted", "grams",
            "stone", "ounces", "oz"
        ],
        "stress": [
            "stress", "stressed", "anxiety", "tension", "anxious", "worried",
            "worry", "pressure", "strain", "overwhelmed", "nervous", "nervousness",
            "stress level", "mental stress", "emotional stress", "burnout",
            "stress score", "relaxation", "calm", "mental health", "wellbeing"
        ]
    ]
    
    private let timeSynonyms: [String: [String]] = [
        "today": [
            "today", "now", "currently", "this day", "present", "right now",
            "at present", "so far today", "today's", "current day", "as of today",
            "till now", "up to now", "at the moment", "presently", "at this time",
            "this very day", "the present day", "nowadays", "in the present",
            "for today", "on this day", "todays", "since midnight"
        ],
        "yesterday": [
            "yesterday", "last day", "previous day", "day before", "1 day ago",
            "one day ago", "a day ago", "the day before", "prior day",
            "the previous day", "24 hours ago", "yesterdays", "yesterday's",
            "the other day", "day prior", "the last day", "past day",
            "the day that was", "the preceding day", "most recent day",
            "the latest day", "just yesterday", "only yesterday", "back yesterday"
        ],
        "last night": [
            "last night", "night", "overnight", "during sleep", "while sleeping",
            "nighttime", "night time", "at night", "during the night",
            "throughout the night", "all night", "past night", "previous night",
            "the other night", "last evening", "yesterday night", "yesterday evening",
            "late yesterday", "after dark", "hours of sleep", "sleeping hours",
            "bedtime", "sleep time", "in bed", "whilst asleep", "sleep period"
        ],
        "this morning": [
            "this morning", "morning", "am", "early today", "earlier today",
            "this am", "today morning", "in the morning", "morning time",
            "mornings", "early hours", "before noon", "dawn", "daybreak",
            "sunrise", "first thing", "early on", "at dawn", "morning hours",
            "start of day", "beginning of day", "waking hours", "after waking",
            "upon waking", "since waking"
        ],
        "this week": [
            "this week", "current week", "weekly", "so far this week",
            "week to date", "wtd", "the week", "present week", "the current week",
            "in this week", "for the week", "throughout the week", "during the week",
            "over the week", "7 days", "past 7 days", "last 7 days",
            "these 7 days", "this weeks", "this week's", "since monday",
            "week so far", "till now this week", "up to now this week", "weekly total"
        ],
        "last week": [
            "last week", "past week", "previous week", "the week before",
            "prior week", "1 week ago", "one week ago", "a week ago",
            "week prior", "the last week", "the past week", "the previous week",
            "7 days ago", "last weeks", "last week's", "the preceding week",
            "most recent week", "latest week", "former week", "earlier week",
            "the other week", "back last week", "during last week", "throughout last week",
            "over last week"
        ],
        "this month": [
            "this month", "current month", "monthly", "so far this month",
            "month to date", "mtd", "the month", "present month",
            "the current month", "in this month", "for the month",
            "throughout the month", "during the month", "over the month",
            "30 days", "past 30 days", "last 30 days", "these 30 days",
            "this months", "this month's", "since the 1st", "month so far",
            "till now this month", "up to now this month", "monthly total"
        ]
    ]
    
    private let qualifierSynonyms: [String: [String]] = [
        "minimum": [
            "minimum", "min", "lowest", "least", "bottom", "bare minimum",
            "minimal", "minimally", "rock bottom", "floor", "base", "baseline",
            "low point", "lower", "smallest", "tiniest", "fewest", "less",
            "lesser", "reduced", "at least", "no less than", "starting from",
            "beginning at", "from", "low", "lows", "worst", "slowest"
        ],
        "maximum": [
            "maximum", "max", "highest", "most", "peak", "top", "maximal",
            "maximally", "ceiling", "upper limit", "high point", "higher",
            "greatest", "largest", "biggest", "best", "record", "all time high",
            "at most", "no more than", "up to", "limit", "cap", "high", "highs",
            "fastest", "extreme", "topmost", "ultimate"
        ],
        "average": [
            "average", "avg", "mean", "typical", "normal", "averaged", "averaging",
            "median", "mid", "middle", "midpoint", "central", "moderate",
            "standard", "regular", "usual", "common", "ordinary", "per day",
            "daily average", "on average", "typically", "normally", "generally",
            "approximately", "around", "about", "roughly"
        ],
        "total": [
            "total", "sum", "overall", "complete", "entire", "totaled", "totaling",
            "all", "all time", "full", "whole", "combined", "cumulative",
            "aggregate", "collectively", "together", "in total", "altogether",
            "grand total", "summation", "net", "gross", "comprehensive",
            "accumulated", "compilation", "tally", "count", "running total"
        ]
    ]
    
    // Pre-compiled regex patterns for better performance
    private let walkingMovementRegex = try! NSRegularExpression(pattern: "\\b(?:walk|walked|walking|stroll|strolled|strolling|hike|hiked|hiking|trek|trekked|trekking|march|marched|marching|wander|wandered|wandering|roam|roamed|roaming|amble|ambled|ambling|saunter|sauntered|sauntering|trudge|trudged|trudging|move|moved|moving|movement)\\b", options: [.caseInsensitive])
    private let distanceRegex = try! NSRegularExpression(pattern: "\\b(?:far|distance|km|kilometers|kilometre|kilometres|mile|miles|meter|meters|metre|metres|feet|ft|yard|yards|yd|long|length|covered|travelled|traveled|route|path|journey|span|range|how far|mileage)\\b", options: [.caseInsensitive])
    private let sleepRegex = try! NSRegularExpression(pattern: "\\b(?:sleep|slept|sleeping|asleep|nap|napped|napping|rest|rested|resting|snooze|snoozed|snoozing|doze|dozed|dozing|slumber|bedtime|night|overnight|bed|zzz)\\b", options: [.caseInsensitive])
    private let sleepQualityRegex = try! NSRegularExpression(pattern: "\\b(?:quality|score|rating|rate|well|badly|good|bad|poor|deep|light|efficiency|grade|rank|analysis|performance|how well)\\b", options: [.caseInsensitive])
    private let heartRegex = try! NSRegularExpression(pattern: "\\b(?:heart|cardiac|cardio|cardiovascular|pulse|beat|beats|beating|bpm|rhythm|ticker)\\b", options: [.caseInsensitive])
    private let caloriesRegex = try! NSRegularExpression(pattern: "\\b(?:calorie|calories|kcal|energy|burn|burned|burnt|burning|expend|expended|consume|consumed|intake|kilojoule|kilojoules|kj|food energy|metabolic|metabolism|fat)\\b", options: [.caseInsensitive])
    private let oxygenRegex = try! NSRegularExpression(pattern: "\\b(?:oxygen|o2|spo2|saturation|sat|blood oxygen|pulse ox|oximeter|oximetry|breathing|respiratory|respiration|air|breathe)\\b", options: [.caseInsensitive])
    private let weightRegex = try! NSRegularExpression(pattern: "\\b(?:weight|weigh|weighing|weighed|kg|kilogram|kilograms|pound|pounds|lbs|lb|body mass|bmi|body weight|mass|scale|heavy|light|stone|gram|grams|ounce|ounces|oz)\\b", options: [.caseInsensitive])
    private let stressRegex = try! NSRegularExpression(pattern: "\\b(?:stress|stressed|stressful|anxiety|anxious|tension|tense|worried|worry|worrying|pressure|pressured|strain|strained|overwhelm|overwhelmed|nervous|nervousness|burnout|mental health|relaxation|relax|calm|peace|peaceful)\\b", options: [.caseInsensitive])
    private let heartRateUnitRegex = try! NSRegularExpression(pattern: "\\b(?:heart\\s+rate|pulse|hr)\\b", options: [.caseInsensitive])
    private let weightUnitRegex = try! NSRegularExpression(pattern: "\\b(?:weight|weigh)\\b", options: [.caseInsensitive])
    private let stepsUnitRegex = try! NSRegularExpression(pattern: "\\bsteps?\\b", options: [.caseInsensitive])
    private let numberRegex = try! NSRegularExpression(pattern: "\\b(\\d+(?:\\.\\d+)?)\\b", options: [])
    private let numberSequenceRegex = try! NSRegularExpression(pattern: "\\b(\\d+(?:\\.\\d+)?)\\b", options: [])
    
    // Pre-compiled synonym patterns for better performance
    private var synonymPatterns: [String: NSRegularExpression] = [:]
    
    override init() {
        super.init()
        // Pre-compile synonym patterns
        for (metric, synonyms) in metricSynonyms {
            let escapedSynonyms = synonyms.map { NSRegularExpression.escapedPattern(for: $0) }
            let pattern = "\\b(?:\(escapedSynonyms.joined(separator: "|")))\\b"
            synonymPatterns[metric] = try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        }
    }
    
    // Pre-compiled unit regex patterns
    private let unitPatterns: [String: NSRegularExpression] = [
        "bpm": try! NSRegularExpression(pattern: "\\b(?:bpm|beats?\\s+per\\s+minute|heart\\s+rate|pulse\\s+rate|hr|heartbeat|heart\\s+beat|pulse|cardiac\\s+rate|beat\\s+rate|rhythm|heart\\s+rhythm|cardiac\\s+rhythm)\\b", options: [.caseInsensitive]),
        "kg": try! NSRegularExpression(pattern: "\\b(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?)\\b", options: [.caseInsensitive]),
        "pounds": try! NSRegularExpression(pattern: "\\b(?:pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight)\\b", options: [.caseInsensitive]),
        "km": try! NSRegularExpression(pattern: "\\b(?:km|kms|kilometer|kilometers|kilometre|kilometres|k\\.?m\\.?|klick|klicks)\\b", options: [.caseInsensitive]),
        "miles": try! NSRegularExpression(pattern: "\\b(?:miles?|mi|mile\\s+distance|mi\\.?|statute\\s+miles?)\\b", options: [.caseInsensitive]),
        "kcal": try! NSRegularExpression(pattern: "\\b(?:kcal|calories?|calorie|cal|cals|kilocalories?|kilocalorie|food\\s+calories?|dietary\\s+calories?|energy|k\\.?cal)\\b", options: [.caseInsensitive]),
        "hours": try! NSRegularExpression(pattern: "\\b(?:hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?)\\b", options: [.caseInsensitive]),
        "minutes": try! NSRegularExpression(pattern: "\\b(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?)\\b", options: [.caseInsensitive]),
        "percent": try! NSRegularExpression(pattern: "\\b(?:percent|%|percentage|pct|pc|per\\s+cent|percentile|blood oxygen|spo2|sp2|spO2)\\b", options: [.caseInsensitive]),
        "count": try! NSRegularExpression(pattern: "\\b(?:steps?|step\\s+count|footsteps?|foot\\s+steps?|paces?|strides?|walk\\s+count|walking\\s+count|number\\s+of\\s+steps?|total\\s+steps?|step\\s+total)\\b", options: [.caseInsensitive]),
        "meters": try! NSRegularExpression(pattern: "\\b(?:meters?|metres?|m|meter\\s+distance|metre\\s+distance|m\\.?)\\b", options: [.caseInsensitive]),
        "feet": try! NSRegularExpression(pattern: "\\b(?:feet|foot|ft|f\\.?t\\.?|')\\b", options: [.caseInsensitive]),
        "seconds": try! NSRegularExpression(pattern: "\\b(?:seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?)\\b", options: [.caseInsensitive]),
        "grams": try! NSRegularExpression(pattern: "\\b(?:grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?)\\b", options: [.caseInsensitive]),
        "liters": try! NSRegularExpression(pattern: "\\b(?:liters?|litres?|l|ltr|ltrs|l\\.?t?r?\\.?s?\\.?)\\b", options: [.caseInsensitive]),
        "degrees": try! NSRegularExpression(pattern: "\\b(?:degrees?|deg|°|degree\\s+celsius|degree\\s+fahrenheit|celsius|fahrenheit)\\b", options: [.caseInsensitive]),
        "score": try! NSRegularExpression(pattern: "\\b(?:score|rating|grade|rank|level|point|points|pts?|value|number)\\b", options: [.caseInsensitive]),
        "distance": try! NSRegularExpression(pattern: "\\b(?:distance|length|span|range|mileage|how\\s+far|travelled|traveled|covered)\\b", options: [.caseInsensitive])
    ]
    
    
    func extractSlots(text: String, intent: String) -> SlotExtractionResult {
        Self.logger.info("🏷️ Extracting slots for intent: \(intent), text: '\(text)'")
        
        var slots: [String: Any] = [:]
        let textLower = text.lowercased()
        
        // Get required slots for this intent
        let requiredSlots = intentSlotTemplates[intent] ?? []
        
        // Pre-process text
        let processedText = preprocessText(textLower)
        
        // Extract each required slot
        for slotName in requiredSlots {
            if let value = extractSingleSlot(processedText: processedText, originalText: textLower, slotName: slotName, intent: intent) {
                slots[slotName] = value
                Self.logger.info("  ✓ Extracted \(slotName): \(value)")
            }
        }
        
        // Add contextual slots
        addContextualSlots(text: textLower, intent: intent, slots: &slots)
        
        // Calculate confidence based on how many slots were extracted
        let confidence: Float = requiredSlots.isEmpty ? 1.0 : Float(slots.count) / Float(requiredSlots.count)
        
        Self.logger.info("🎯 Final slots: \(slots) (confidence: \(String(format: "%.2f", confidence)))")
        
        return SlotExtractionResult(slots: slots, confidence: confidence)
    }
    
    private func preprocessText(_ text: String) -> String {
        var processed = text
        
        // Normalize common variations (from Kotlin implementation)
        processed = processed.replacingOccurrences(of: "\\bhow\\s+much\\s+did\\s+i\\s+walk", with: "walking distance", options: .regularExpression)
        processed = processed.replacingOccurrences(of: "\\bhow\\s+many\\s+steps", with: "steps", options: .regularExpression)
        processed = processed.replacingOccurrences(of: "\\bhow\\s+far\\s+did\\s+i\\s+walk", with: "walking distance", options: .regularExpression)
        processed = processed.replacingOccurrences(of: "\\bwhat\\s+is\\s+my", with: "my", options: .regularExpression)
        processed = processed.replacingOccurrences(of: "\\bshow\\s+me\\s+my", with: "my", options: .regularExpression)
        
        return processed
    }
    
    private func extractSingleSlot(processedText: String, originalText: String, slotName: String, intent: String) -> Any? {
        switch slotName {
        case "metric":
            return extractMetric(processedText: processedText, originalText: originalText)
        case "time_ref":
            return extractTimeRef(text: originalText)
        case "unit":
            return extractUnit(text: originalText)
        case "qualifier":
            return extractQualifier(text: originalText)
        case "identifier":
            return extractIdentifier(text: originalText)
        case "threshold":
            return extractThreshold(text: originalText)
        case "target":
            return extractTarget(text: originalText)
        case "value":
            return extractValue(text: originalText, intent: intent)
        case "feature":
            return extractFeature(text: originalText)
        case "state":
            return extractState(text: originalText)
        case "action":
            return extractAction(text: originalText)
        case "tool":
            return extractTool(text: originalText)
        case "activity_type":
            return extractActivityType(text: originalText)
        case "app":
            return extractApp(text: originalText)
        case "contact":
            return extractContact(text: originalText)
        case "location":
            return extractLocation(text: originalText)
        case "attribute":
            return extractAttribute(text: originalText)
        case "type":
            return extractType(text: originalText)
        case "period":
            return extractPeriod(text: originalText)
        case "event_type":
            return extractEventType(text: originalText)
        case "message":
            return "Sorry, please say again"
        default:
            return nil
        }
    }
    
    
    private func extractMetric(processedText: String, originalText: String) -> String? {
        // Direct synonym matching on processed text first
        for (metric, pattern) in synonymPatterns {
            if pattern.numberOfMatches(in: processedText, options: [], range: NSRange(location: 0, length: processedText.count)) > 0 {
                Self.logger.info("  📏 Found metric '\(metric)' via synonym pattern in processed text")
                return metric
            }
        }
        
        // Fallback to original text
        for (metric, pattern) in synonymPatterns {
            if pattern.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
                Self.logger.info("  📏 Found metric '\(metric)' via synonym pattern in original text")
                return metric
            }
        }
        
        // Context-based inference with expanded patterns
        
        // Walking/Movement context
        if walkingMovementRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return distanceRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 ? "distance" : "steps"
        }
        
        // Sleep context
        if sleepRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return sleepQualityRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 ? "sleep score" : "sleep"
        }
        
        // Heart/Cardio context
        if heartRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return "heart rate"
        }
        
        // Calories/Energy context
        if caloriesRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return "calories"
        }
        
        // Oxygen/Breathing context
        if oxygenRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return "spo2"
        }
        
        // Weight/Body Mass context
        if weightRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return "weight"
        }
        
        // Stress/Mental Health context
        if stressRegex.numberOfMatches(in: originalText, options: [], range: NSRange(location: 0, length: originalText.count)) > 0 {
            return "stress"
        }
        
        return nil
    }
    
    private func extractTimeRef(text: String) -> String? {
        let timePatterns: [String: String] = [
            "last night": "\\blast\\s+night\\b|\\bduring\\s+(?:the\\s+)?night\\b|\\bovernight\\b|\\bnight\\s+time\\b|\\bnighttime\\b|\\bat\\s+night\\b|\\bthrough(?:out)?\\s+(?:the\\s+)?night\\b|\\ball\\s+night\\b|\\bpast\\s+night\\b|\\bprevious\\s+night\\b|\\bthe\\s+other\\s+night\\b|\\blast\\s+evening\\b|\\byesterday\\s+night\\b|\\byesterday\\s+evening\\b|\\blate\\s+yesterday\\b|\\bafter\\s+dark\\b|\\bwhile\\s+(?:I\\s+)?sleep(?:ing)?\\b|\\bduring\\s+(?:my\\s+)?sleep\\b|\\bsleep(?:ing)?\\s+(?:hours|time|period)\\b|\\bin\\s+bed\\b|\\bwhilst\\s+asleep\\b|\\bwhen\\s+(?:I\\s+)?(?:was\\s+)?asleep\\b|\\bbedtime\\b",
            
            "yesterday": "\\byesterday\\b(?!\\s+(?:night|evening))|\\blast\\s+day\\b|\\bprevious\\s+day\\b|\\bday\\s+before\\b|\\b1\\s+day\\s+ago\\b|\\bone\\s+day\\s+ago\\b|\\ba\\s+day\\s+ago\\b|\\bthe\\s+day\\s+before\\b|\\bprior\\s+day\\b|\\bthe\\s+previous\\s+day\\b|\\b24\\s+hours\\s+ago\\b|\\byesterdays\\b|\\byesterday's\\b|\\bthe\\s+other\\s+day\\b|\\bday\\s+prior\\b|\\bthe\\s+last\\s+day\\b|\\bpast\\s+day\\b|\\bthe\\s+day\\s+that\\s+was\\b|\\bthe\\s+preceding\\s+day\\b|\\bmost\\s+recent\\s+day\\b|\\bthe\\s+latest\\s+day\\b|\\bjust\\s+yesterday\\b|\\bonly\\s+yesterday\\b|\\bback\\s+yesterday\\b",
            
            "yesterday morning": "\\byesterday\\s+morning\\b|\\bmorning\\s+yesterday\\b|\\byesterday\\s+am\\b|\\byesterday\\s+in\\s+the\\s+morning\\b|\\bthe\\s+morning\\s+yesterday\\b|\\bearly\\s+yesterday\\b|\\byesterday\\s+early\\b|\\byesterday\\s+dawn\\b|\\byesterday\\s+daybreak\\b|\\byesterday\\s+sunrise\\b|\\byesterday\\s+before\\s+noon\\b",
            
            "yesterday afternoon": "\\byesterday\\s+afternoon\\b|\\bafternoon\\s+yesterday\\b|\\byesterday\\s+pm\\b|\\byesterday\\s+in\\s+the\\s+afternoon\\b|\\bthe\\s+afternoon\\s+yesterday\\b|\\byesterday\\s+after\\s+noon\\b|\\byesterday\\s+midday\\b|\\byesterday\\s+mid-day\\b|\\byesterday\\s+noon\\b|\\byesterday\\s+lunch\\s+time\\b",
            
            "yesterday evening": "\\byesterday\\s+evening\\b|\\bevening\\s+yesterday\\b|\\byesterday\\s+in\\s+the\\s+evening\\b|\\bthe\\s+evening\\s+yesterday\\b|\\byesterday\\s+at\\s+night\\b|\\byesterday\\s+dusk\\b|\\byesterday\\s+twilight\\b|\\byesterday\\s+sunset\\b|\\byesterday\\s+after\\s+dark\\b|\\blast\\s+evening\\b",
            
            "today": "\\btoday\\b|\\bnow\\b|\\bcurrently\\b|\\bthis\\s+day\\b|\\bpresent\\b|\\bright\\s+now\\b|\\bat\\s+present\\b|\\bso\\s+far\\s+today\\b|\\btoday's\\b|\\bcurrent\\s+day\\b|\\bas\\s+of\\s+today\\b|\\btill\\s+now\\b|\\bup\\s+to\\s+now\\b|\\bat\\s+the\\s+moment\\b|\\bpresently\\b|\\bat\\s+this\\s+time\\b|\\bthis\\s+very\\s+day\\b|\\bthe\\s+present\\s+day\\b|\\bnowadays\\b|\\bin\\s+the\\s+present\\b|\\bfor\\s+today\\b|\\bon\\s+this\\s+day\\b|\\btodays\\b|\\bsince\\s+midnight\\b",
            
            "this morning": "\\bthis\\s+morning\\b|\\bmorning\\b|\\bam\\b|\\bearly\\s+today\\b|\\bearlier\\s+today\\b|\\bthis\\s+am\\b|\\btoday\\s+morning\\b|\\bin\\s+the\\s+morning\\b|\\bmorning\\s+time\\b|\\bmornings\\b|\\bearly\\s+hours\\b|\\bbefore\\s+noon\\b|\\bdawn\\b|\\bdaybreak\\b|\\bsunrise\\b|\\bfirst\\s+thing\\b|\\bearly\\s+on\\b|\\bat\\s+dawn\\b|\\bmorning\\s+hours\\b|\\bstart\\s+of\\s+day\\b|\\bbeginning\\s+of\\s+day\\b|\\bwaking\\s+hours\\b|\\bafter\\s+waking\\b|\\bupon\\s+waking\\b|\\bsince\\s+waking\\b",
            
            "this afternoon": "\\bthis\\s+afternoon\\b|\\bafternoon\\b|\\bpm\\b|\\bafter\\s+noon\\b|\\bmidday\\b|\\bmid-day\\b|\\bnoon\\s+time\\b|\\blunch\\s+time\\b|\\bpost\\s+meridiem\\b|\\bin\\s+the\\s+afternoon\\b|\\bafternoon\\s+time\\b|\\bafternoons\\b|\\bmid\\s+day\\b|\\bafter\\s+lunch\\b|\\bthis\\s+pm\\b|\\btoday\\s+afternoon\\b",
            
            "this evening": "\\bthis\\s+evening\\b|\\bevening\\b|\\btonight\\b|\\bat\\s+night\\b|\\bdusk\\b|\\btwilight\\b|\\bsunset\\b|\\bafter\\s+dark\\b|\\bnightfall\\b|\\bin\\s+the\\s+evening\\b|\\bevening\\s+time\\b|\\bevenings\\b|\\bthis\\s+night\\b|\\btoday\\s+evening\\b|\\btoday\\s+night\\b|\\blate\\s+today\\b|\\bend\\s+of\\s+day\\b",
            
            "this week": "\\bthis\\s+week\\b|\\bcurrent\\s+week\\b|\\bweekly\\b|\\bso\\s+far\\s+this\\s+week\\b|\\bweek\\s+to\\s+date\\b|\\bwtd\\b|\\bthe\\s+week\\b|\\bpresent\\s+week\\b|\\bthe\\s+current\\s+week\\b|\\bin\\s+this\\s+week\\b|\\bfor\\s+the\\s+week\\b|\\bthroughout\\s+the\\s+week\\b|\\bduring\\s+the\\s+week\\b|\\bover\\s+the\\s+week\\b|\\b7\\s+days\\b|\\bpast\\s+7\\s+days\\b|\\blast\\s+7\\s+days\\b|\\bthese\\s+7\\s+days\\b|\\bthis\\s+weeks\\b|\\bthis\\s+week's\\b|\\bsince\\s+monday\\b|\\bweek\\s+so\\s+far\\b|\\btill\\s+now\\s+this\\s+week\\b|\\bup\\s+to\\s+now\\s+this\\s+week\\b|\\bweekly\\s+total\\b",
            
            "last week": "\\blast\\s+week\\b|\\bpast\\s+week\\b|\\bprevious\\s+week\\b|\\bthe\\s+week\\s+before\\b|\\bprior\\s+week\\b|\\b1\\s+week\\s+ago\\b|\\bone\\s+week\\s+ago\\b|\\ba\\s+week\\s+ago\\b|\\bweek\\s+prior\\b|\\bthe\\s+last\\s+week\\b|\\bthe\\s+past\\s+week\\b|\\bthe\\s+previous\\s+week\\b|\\b7\\s+days\\s+ago\\b|\\blast\\s+weeks\\b|\\blast\\s+week's\\b|\\bthe\\s+preceding\\s+week\\b|\\bmost\\s+recent\\s+week\\b|\\blatest\\s+week\\b|\\bformer\\s+week\\b|\\bearlier\\s+week\\b|\\bthe\\s+other\\s+week\\b|\\bback\\s+last\\s+week\\b|\\bduring\\s+last\\s+week\\b|\\bthroughout\\s+last\\s+week\\b|\\bover\\s+last\\s+week\\b",
            
            "this month": "\\bthis\\s+month\\b|\\bcurrent\\s+month\\b|\\bmonthly\\b|\\bso\\s+far\\s+this\\s+month\\b|\\bmonth\\s+to\\s+date\\b|\\bmtd\\b|\\bthe\\s+month\\b|\\bpresent\\s+month\\b|\\bthe\\s+current\\s+month\\b|\\bin\\s+this\\s+month\\b|\\bfor\\s+the\\s+month\\b|\\bthroughout\\s+the\\s+month\\b|\\bduring\\s+the\\s+month\\b|\\bover\\s+the\\s+month\\b|\\b30\\s+days\\b|\\bpast\\s+30\\s+days\\b|\\blast\\s+30\\s+days\\b|\\bthese\\s+30\\s+days\\b|\\bthis\\s+months\\b|\\bthis\\s+month's\\b|\\bsince\\s+the\\s+1st\\b|\\bmonth\\s+so\\s+far\\b|\\btill\\s+now\\s+this\\s+month\\b|\\bup\\s+to\\s+now\\s+this\\s+month\\b|\\bmonthly\\s+total\\b"
        ]
        
        for (timeRef, pattern) in timePatterns {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return timeRef
            }
        }
        
        return nil
    }
    
    private func extractQualifier(text: String) -> String? {
        let qualifierPatterns: [String: String] = [
            "minimum": "\\b(?:minimum|min|lowest|least|bottom|bare minimum|minimal|minimally|rock bottom|floor|base|baseline|low point|lower|smallest|tiniest|fewest|less|lesser|reduced|at least|no less than|starting from|beginning at|from|low|lows|worst|slowest)\\b",
            "maximum": "\\b(?:maximum|max|highest|most|peak|top|maximal|maximally|ceiling|upper limit|high point|higher|greatest|largest|biggest|best|record|all time high|at most|no more than|up to|limit|cap|high|highs|fastest|extreme|topmost|ultimate)\\b",
            "average": "\\b(?:average|avg|mean|typical|normal|averaged|averaging|median|mid|middle|midpoint|central|moderate|standard|regular|usual|common|ordinary|per day|daily average|on average|typically|normally|generally|approximately|around|about|roughly)\\b",
            "total": "\\b(?:total|sum|overall|complete|entire|totaled|totaling|all|all time|full|whole|combined|cumulative|aggregate|collectively|together|in total|altogether|grand total|summation|net|gross|comprehensive|accumulated|compilation|tally|count|running total)\\b"
        ]
        
        for (qualifier, pattern) in qualifierPatterns {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return qualifier
            }
        }
        
        return "today"
    }
    
    private func extractIdentifier(text: String) -> String? {
        let identifierPatterns: [String: String] = [
            "minimum": "\\b(?:minimum|min|lowest|least|bottom|bare minimum|minimal|minimally|rock bottom|floor|base|baseline|low point|lower|smallest|tiniest|fewest|less|lesser|reduced|at least|no less than|starting from|beginning at|from|low|lows|worst|slowest)\\b",
            "maximum": "\\b(?:maximum|max|highest|most|peak|top|maximal|maximally|ceiling|upper limit|high point|higher|greatest|largest|biggest|best|record|all time high|at most|no more than|up to|limit|cap|high|highs|fastest|extreme|topmost|ultimate)\\b",
            "average": "\\b(?:average|avg|mean|typical|normal|averaged|averaging|median|mid|middle|midpoint|central|moderate|standard|regular|usual|common|ordinary|per day|daily average|on average|typically|normally|generally|approximately|around|about|roughly)\\b",
            "total": "\\b(?:total|sum|overall|complete|entire|totaled|totaling|all|all time|full|whole|combined|cumulative|aggregate|collectively|together|in total|altogether|grand total|summation|net|gross|comprehensive|accumulated|compilation|tally|count|running total)\\b"
        ]
        
        for (identifier, pattern) in identifierPatterns {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return identifier
            }
        }
        
        // Default to average if no specific identifier is found
        return "average"
    }
    
    private func extractUnit(text: String) -> String? {
        // Try context-based unit inference first
        if heartRateUnitRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "bpm"
        }
        
        if weightUnitRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "kg"
        }
        
        if stepsUnitRegex.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
            return "count"
        }
        
        // Check pre-compiled unit patterns
        for (unit, pattern) in unitPatterns {
            if pattern.numberOfMatches(in: text, options: [], range: NSRange(location: 0, length: text.count)) > 0 {
                return unit
            }
        }
        
        return nil
    }
    
    
    private func extractThreshold(text: String) -> Int? {
        // Look for numbers in context
        let numberPatterns = [
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:bpm|kg|km|miles?|percent|%|hours?|minutes?|calories?|kcal|steps?|pounds?|lbs?)\\b",
            "\\b(?:above|over|exceeds?|higher\\s+than|more\\s+than|greater\\s+than|beyond|surpass|surpasses|surpassed|passing|exceed|exceeding|exceeded|cross|crosses|crossed|crossing|top|tops|topped|topping|beat|beats|beaten|beating)\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:below|under|less\\s+than|lower\\s+than|beneath|underneath|drop|drops|dropped|dropping|fall|falls|fell|falling|decrease|decreases|decreased|decreasing|reduce|reduces|reduced|reducing|decline|declines|declined|declining|go\\s+down|goes\\s+down|went\\s+down|going\\s+down)\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:when|if|whenever|once|after|before|upon|as\\s+soon\\s+as|the\\s+moment|the\\s+instant|immediately\\s+when|right\\s+when|just\\s+when|exactly\\s+when).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:reaches?|reached|reaching|gets?\\s+to|got\\s+to|getting\\s+to|arrives?\\s+at|arrived\\s+at|arriving\\s+at|hits?|hit|hitting|touches?|touched|touching|meets?|met|meeting|achieves?|achieved|achieving|attains?|attained|attaining)\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:threshold|limit|cap|ceiling|maximum|max|minimum|min|floor|baseline|target|goal|aim|objective|mark|point|level|value|number|figure|amount|quantity|measurement|reading).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?).*?(?:threshold|limit|cap|ceiling|maximum|max|minimum|min|floor|baseline|target|goal|aim|objective|mark|point|level|value|number|figure|amount|quantity|measurement|reading)\\b",
            "\\b(?:set|setting|configured|configure|define|defined|defining|establish|established|establishing|create|created|creating|make|made|making|put|putting|place|placed|placing|fix|fixed|fixing|determine|determined|determining|specify|specified|specifying).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?).*?(?:set|setting|configured|configure|define|defined|defining|establish|established|establishing|create|created|creating|make|made|making|put|putting|place|placed|placing|fix|fixed|fixing|determine|determined|determining|specify|specified|specifying)\\b",
            "\\b(?:warn|warning|warned|alert|alerts|alerted|alerting|notify|notifies|notified|notifying|inform|informs|informed|informing|tell|tells|told|telling|remind|reminds|reminded|reminding).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?).*?(?:warn|warning|warned|alert|alerts|alerted|alerting|notify|notifies|notified|notifying|inform|informs|informed|informing|tell|tells|told|telling|remind|reminds|reminded|reminding)\\b",
            "\\b(?:trigger|triggers|triggered|triggering|activate|activates|activated|activating|start|starts|started|starting|begin|begins|began|beginning|initiate|initiates|initiated|initiating|launch|launches|launched|launching|fire|fires|fired|firing).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?).*?(?:trigger|triggers|triggered|triggering|activate|activates|activated|activating|start|starts|started|starting|begin|begins|began|beginning|initiate|initiates|initiated|initiating|launch|launches|launched|launching|fire|fires|fired|firing)\\b"
        ]
        
        for pattern in numberPatterns {
            if let range = text.range(of: pattern, options: .regularExpression) {
                let matchedText = String(text[range])
                let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                if let match = matches.first {
                    let numberRange = Range(match.range, in: matchedText)!
                    let numberString = String(matchedText[numberRange])
                    if let number = Double(numberString) {
                        return Int(number)
                    }
                }
            }
        }
        
        // Fallback to any number
        let matches = numberRegex.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))
        if let match = matches.first {
            let numberRange = Range(match.range, in: text)!
            let numberString = String(text[numberRange])
            if let number = Double(numberString) {
                return Int(number)
            }
        }
        
        return nil
    }
    
    private func extractTarget(text: String) -> Int? {
        let goalPatterns = [
            "\\b(?:goal|target|aim|objective|purpose|intention|plan|ambition|aspiration|desire|want|wish|hope|dream|vision|mission).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?).*?(?:goal|target|aim|objective|purpose|intention|plan|ambition|aspiration|desire|want|wish|hope|dream|vision|mission)\\b",
            "\\b(?:set|setting|change|changing|update|updating|modify|modifying|alter|altering|adjust|adjusting|configure|configuring|establish|establishing|create|creating|make|making|put|putting|place|placing|fix|fixing|determine|determining|specify|specifying|define|defining).*?(?:to|at|for|as|with|into|onto|upon|on|in)\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:to|at|for|as|with|into|onto|upon|on|in)\\s*(\\d+(?:\\.\\d+)?).*?(?:set|setting|change|changing|update|updating|modify|modifying|alter|altering|adjust|adjusting|configure|configuring|establish|establishing|create|creating|make|making|put|putting|place|placing|fix|fixing|determine|determining|specify|specifying|define|defining)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:steps?|step\\s+count|footsteps?|foot\\s+steps?|paces?|strides?|walk\\s+count|walking\\s+count|number\\s+of\\s+steps?|total\\s+steps?|step\\s+total)\\b",
            "\\b(?:steps?|step\\s+count|footsteps?|foot\\s+steps?|paces?|strides?|walk\\s+count|walking\\s+count|number\\s+of\\s+steps?|total\\s+steps?|step\\s+total).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?|pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight|grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?|stone|ounces?|ounce|oz)\\b",
            "\\b(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?|pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight|grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?|stone|ounces?|ounce|oz).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:km|kms|kilometer|kilometers|kilometre|kilometres|k\\.?m\\.?|klick|klicks|miles?|mi|mile\\s+distance|mi\\.?|statute\\s+miles?|meters?|metres?|m|meter\\s+distance|metre\\s+distance|m\\.?|feet|foot|ft|f\\.?t\\.?|')\\b",
            "\\b(?:km|kms|kilometer|kilometers|kilometre|kilometres|k\\.?m\\.?|klick|klicks|miles?|mi|mile\\s+distance|mi\\.?|statute\\s+miles?|meters?|metres?|m|meter\\s+distance|metre\\s+distance|m\\.?|feet|foot|ft|f\\.?t\\.?|').*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|minutes?|mins?|min|minute\\s+duration|min\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?)\\b",
            "\\b(?:hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|minutes?|mins?|min|minute\\s+duration|min\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:kcal|calories?|calorie|cal|cals|kilocalories?|kilocalorie|food\\s+calories?|dietary\\s+calories?|energy|k\\.?cal)\\b",
            "\\b(?:kcal|calories?|calorie|cal|cals|kilocalories?|kilocalorie|food\\s+calories?|dietary\\s+calories?|energy|k\\.?cal).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:bpm|beats?\\s+per\\s+minute|heart\\s+rate|pulse\\s+rate|hr|heartbeat|heart\\s+beat|pulse|cardiac\\s+rate|beat\\s+rate|rhythm|heart\\s+rhythm|cardiac\\s+rhythm)\\b",
            "\\b(?:bpm|beats?\\s+per\\s+minute|heart\\s+rate|pulse\\s+rate|hr|heartbeat|heart\\s+beat|pulse|cardiac\\s+rate|beat\\s+rate|rhythm|heart\\s+rhythm|cardiac\\s+rhythm).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:percent|%|percentage|pct|pc|per\\s+cent|percentile|blood\\s+oxygen|spo2|sp2|spO2)\\b",
            "\\b(?:percent|%|percentage|pct|pc|per\\s+cent|percentile|blood\\s+oxygen|spo2|sp2|spO2).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:reach|reaching|reached|get|getting|got|achieve|achieving|achieved|attain|attaining|attained|hit|hitting|meet|meeting|met|arrive|arriving|arrived|touch|touching|touched|obtain|obtaining|obtained|acquire|acquiring|acquired|gain|gaining|gained|earn|earning|earned|secure|securing|secured|accomplish|accomplishing|accomplished|complete|completing|completed|finish|finishing|finished).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?).*?(?:reach|reaching|reached|get|getting|got|achieve|achieving|achieved|attain|attaining|attained|hit|hitting|meet|meeting|met|arrive|arriving|arrived|touch|touching|touched|obtain|obtaining|obtained|acquire|acquiring|acquired|gain|gaining|gained|earn|earning|earned|secure|securing|secured|accomplish|accomplishing|accomplished|complete|completing|completed|finish|finishing|finished)\\b",
            "\\b(?:want|wanting|wanted|need|needing|needed|require|requiring|required|expect|expecting|expected|hope|hoping|hoped|wish|wishing|wished|desire|desiring|desired|aim\\s+for|aiming\\s+for|aimed\\s+for|strive\\s+for|striving\\s+for|strived\\s+for|work\\s+towards|working\\s+towards|worked\\s+towards|shoot\\s+for|shooting\\s+for|shot\\s+for).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?).*?(?:want|wanting|wanted|need|needing|needed|require|requiring|required|expect|expecting|expected|hope|hoping|hoped|wish|wishing|wished|desire|desiring|desired|aim\\s+for|aiming\\s+for|aimed\\s+for|strive\\s+for|striving\\s+for|strived\\s+for|work\\s+towards|working\\s+towards|worked\\s+towards|shoot\\s+for|shooting\\s+for|shot\\s+for)\\b",
            "\\b(?:increase|increasing|increased|boost|boosting|boosted|raise|raising|raised|lift|lifting|lifted|elevate|elevating|elevated|enhance|enhancing|enhanced|improve|improving|improved|better|bettering|bettered|upgrade|upgrading|upgraded|advance|advancing|advanced|progress|progressing|progressed|grow|growing|grew|expand|expanding|expanded|extend|extending|extended|amplify|amplifying|amplified|maximize|maximizing|maximized|optimize|optimizing|optimized).*?(?:to|by|up\\s+to|until|till|reaching|hitting|getting\\s+to|arriving\\s+at)\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:to|by|up\\s+to|until|till|reaching|hitting|getting\\s+to|arriving\\s+at)\\s*(\\d+(?:\\.\\d+)?).*?(?:increase|increasing|increased|boost|boosting|boosted|raise|raising|raised|lift|lifting|lifted|elevate|elevating|elevated|enhance|enhancing|enhanced|improve|improving|improved|better|bettering|bettered|upgrade|upgrading|upgraded|advance|advancing|advanced|progress|progressing|progressed|grow|growing|grew|expand|expanding|expanded|extend|extending|extended|amplify|amplifying|amplified|maximize|maximizing|maximized|optimize|optimizing|optimized)\\b"
        ]
        
        // Try each pattern
        for pattern in goalPatterns {
            if let range = text.range(of: pattern, options: .regularExpression) {
                let matchedText = String(text[range])
                let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                if let match = matches.first {
                    let numberRange = Range(match.range, in: matchedText)!
                    let numberString = String(matchedText[numberRange])
                    if let number = Double(numberString) {
                        return Int(number)
                    }
                }
            }
        }
        
        // Fallback: extract any number from the text
        let matches = numberRegex.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))
        if let match = matches.first {
            let numberRange = Range(match.range, in: text)!
            let numberString = String(text[numberRange])
            if let number = Double(numberString) {
                return Int(number)
            }
        }
        
        return nil
    }
    
    // MARK: - Time Normalization Helper
    
    /// Normalizes various time formats to 24-hour HH:MM format
    /// Examples:
    /// - "730" -> "07:30"
    /// - "730 pm" -> "19:30"  
    /// - "1030 pm" -> "22:30"
    /// - "2230" -> "22:30"
    /// - "7:30 am" -> "07:30"
    /// - "12:00 pm" -> "12:00"
    /// - "12:00 am" -> "00:00"
    private func normalizeTimeFormat(_ timeString: String) -> String {
        let cleanTime = timeString.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        
        // Pattern for times like "730", "1030", "2230" (3-4 digits)
        if let regex = try? NSRegularExpression(pattern: "^(\\d{3,4})$"),
           let match = regex.firstMatch(in: cleanTime, range: NSRange(cleanTime.startIndex..., in: cleanTime)) {
            if let range = Range(match.range(at: 1), in: cleanTime) {
                let digits = String(cleanTime[range])
                if digits.count == 3 {
                    // e.g., "730" -> "07:30"
                    let hour = String(digits.prefix(1))
                    let minute = String(digits.suffix(2))
                    return String(format: "%02d:%@", Int(hour) ?? 0, minute)
                } else if digits.count == 4 {
                    // e.g., "2230" -> "22:30"
                    let hour = String(digits.prefix(2))
                    let minute = String(digits.suffix(2))
                    return "\(hour):\(minute)"
                }
            }
        }
        
        // Pattern for times like "730 pm", "1030 am"
        let amPmPattern = "^(\\d{1,4})\\s*(am|pm)$"
        if let regex = try? NSRegularExpression(pattern: amPmPattern),
           let match = regex.firstMatch(in: cleanTime, range: NSRange(cleanTime.startIndex..., in: cleanTime)) {
            if let timeRange = Range(match.range(at: 1), in: cleanTime),
               let amPmRange = Range(match.range(at: 2), in: cleanTime) {
                let timeDigits = String(cleanTime[timeRange])
                let amPm = String(cleanTime[amPmRange])
                
                var hour: Int = 0
                var minute: Int = 0
                
                if timeDigits.count <= 2 {
                    // e.g., "7 pm" -> hour = 7, minute = 0
                    hour = Int(timeDigits) ?? 0
                    minute = 0
                } else if timeDigits.count == 3 {
                    // e.g., "730 pm" -> hour = 7, minute = 30
                    hour = Int(String(timeDigits.prefix(1))) ?? 0
                    minute = Int(String(timeDigits.suffix(2))) ?? 0
                } else if timeDigits.count == 4 {
                    // e.g., "1030 pm" -> hour = 10, minute = 30
                    hour = Int(String(timeDigits.prefix(2))) ?? 0
                    minute = Int(String(timeDigits.suffix(2))) ?? 0
                }
                
                // Convert to 24-hour format
                if amPm == "pm" && hour != 12 {
                    hour += 12
                } else if amPm == "am" && hour == 12 {
                    hour = 0
                }
                
                return String(format: "%02d:%02d", hour, minute)
            }
        }
        
        // Pattern for times with colon like "7:30 pm", "10:30 am"
        let colonAmPmPattern = "^(\\d{1,2}):(\\d{2})\\s*(am|pm)$"
        if let regex = try? NSRegularExpression(pattern: colonAmPmPattern),
           let match = regex.firstMatch(in: cleanTime, range: NSRange(cleanTime.startIndex..., in: cleanTime)) {
            if let hourRange = Range(match.range(at: 1), in: cleanTime),
               let minuteRange = Range(match.range(at: 2), in: cleanTime),
               let amPmRange = Range(match.range(at: 3), in: cleanTime) {
                var hour = Int(cleanTime[hourRange]) ?? 0
                let minute = Int(cleanTime[minuteRange]) ?? 0
                let amPm = String(cleanTime[amPmRange])
                
                // Convert to 24-hour format
                if amPm == "pm" && hour != 12 {
                    hour += 12
                } else if amPm == "am" && hour == 12 {
                    hour = 0
                }
                
                return String(format: "%02d:%02d", hour, minute)
            }
        }
        
        // Pattern for 24-hour format like "22:30", "07:30"
        let twentyFourHourPattern = "^(\\d{1,2}):(\\d{2})$"
        if let regex = try? NSRegularExpression(pattern: twentyFourHourPattern),
           let match = regex.firstMatch(in: cleanTime, range: NSRange(cleanTime.startIndex..., in: cleanTime)) {
            if let hourRange = Range(match.range(at: 1), in: cleanTime),
               let minuteRange = Range(match.range(at: 2), in: cleanTime) {
                let hour = Int(cleanTime[hourRange]) ?? 0
                let minute = Int(cleanTime[minuteRange]) ?? 0
                return String(format: "%02d:%02d", hour, minute)
            }
        }
        
        // Return original string if no pattern matches
        return timeString
    }
    
    
    private func extractValue(text: String, intent: String) -> Any? {
        switch intent {
        case "LogEvent":
            // Weight patterns
            let weightPatterns = [
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:stone|ounces?|ounce|oz)\\b",
                "\\b(?:weigh|weight|weighing|weighed|mass|body\\s+weight|body\\s+mass|scale|heavy|light).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:weigh|weight|weighing|weighed|mass|body\\s+weight|body\\s+mass|scale|heavy|light)\\b",
                "\\b(?:i\\s+)?(?:am|was|have|had|got|get|getting|became|become|becoming|turned|turn|turning|went|go|going|reached|reach|reaching|hit|hitting|measured|measure|measuring|recorded|record|recording|logged|log|logging|entered|enter|entering|input|inputted|inputting|typed|type|typing|added|add|adding).*?(\\d+(?:\\.\\d+)?)\\s*(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?|pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight|grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?|stone|ounces?|ounce|oz)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?|pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight|grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?|stone|ounces?|ounce|oz).*?(?:i\\s+)?(?:am|was|have|had|got|get|getting|became|become|becoming|turned|turn|turning|went|go|going|reached|reach|reaching|hit|hitting|measured|measure|measuring|recorded|record|recording|logged|log|logging|entered|enter|entering|input|inputted|inputting|typed|type|typing|added|add|adding)\\b",
                "\\b(?:my|current|new|latest|recent|today's|this\\s+morning's|yesterday's|last\\s+night's)\\s*(?:weight|mass|body\\s+weight|body\\s+mass).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:my|current|new|latest|recent|today's|this\\s+morning's|yesterday's|last\\s+night's)\\s*(?:weight|mass|body\\s+weight|body\\s+mass)\\b",
                "\\b(?:scale|scales|weighing\\s+scale|bathroom\\s+scale|digital\\s+scale|weight\\s+scale).*?(?:shows?|showed|showing|says?|said|saying|reads?|read|reading|displays?|displayed|displaying|indicates?|indicated|indicating|reports?|reported|reporting|gives?|gave|giving|tells?|told|telling).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:scale|scales|weighing\\s+scale|bathroom\\s+scale|digital\\s+scale|weight\\s+scale).*?(?:shows?|showed|showing|says?|said|saying|reads?|read|reading|displays?|displayed|displaying|indicates?|indicated|indicating|reports?|reported|reporting|gives?|gave|giving|tells?|told|telling)\\b"
            ]
            
            for pattern in weightPatterns {
                if let range = text.range(of: pattern, options: .regularExpression) {
                    let matchedText = String(text[range])
                    let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                    if let match = matches.first {
                        let numberRange = Range(match.range, in: matchedText)!
                        let numberString = String(matchedText[numberRange])
                        if let number = Double(numberString) {
                            return number
                        }
                    }
                }
            }
            
            // General number extraction for other log events
            let generalPatterns = [
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:ml|milliliters?|millilitres?|l|liters?|litres?|ltr|ltrs|l\\.?t?r?\\.?s?\\.?|cups?|glasses?|bottles?|fl\\s+oz|fluid\\s+ounces?|pints?|quarts?|gallons?)\\b",
                "\\b(?:drank|drink|drinking|drunk|consumed|consume|consuming|had|have|having|took|take|taking|ingested|ingest|ingesting|swallowed|swallow|swallowing).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:drank|drink|drinking|drunk|consumed|consume|consuming|had|have|having|took|take|taking|ingested|ingest|ingesting|swallowed|swallow|swallowing)\\b",
                "\\b(?:amount|quantity|volume|dose|serving|portion|measurement|count|number|total|sum).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:amount|quantity|volume|dose|serving|portion|measurement|count|number|total|sum)\\b"
            ]
            
            for pattern in generalPatterns {
                if let range = text.range(of: pattern, options: .regularExpression) {
                    let matchedText = String(text[range])
                    let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                    if let match = matches.first {
                        let numberRange = Range(match.range, in: matchedText)!
                        let numberString = String(matchedText[numberRange])
                        if let number = Double(numberString) {
                            return number
                        }
                    }
                }
            }
            
        case "TimerStopwatch":
            // Time patterns with comprehensive coverage
            let timePatterns = [
                "\\b(\\d{1,2}(?::\\d{2})?(?:\\s*[ap]m)?)\\b",
                "\\b(\\d+)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?)\\b",
                "\\b(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?).*?(\\d+)\\b",
                "\\b(\\d+)\\s*(?:hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?)\\b",
                "\\b(?:hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?).*?(\\d+)\\b",
                "\\b(\\d+)\\s*(?:seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?)\\b",
                "\\b(?:seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?).*?(\\d+)\\b",
                "\\b(\\d+)\\s*(?:and\\s+)?(?:\\d+)?\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?)\\b",
                "\\b(\\d+)\\s*(?:hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?)\\s*(?:and\\s+)?(?:\\d+)?\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?)\\b",
                "\\b(?:for|during|in|after|within|over|under|about|around|approximately|roughly|exactly|precisely|just|only|at\\s+least|at\\s+most|more\\s+than|less\\s+than|up\\s+to|down\\s+to)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?|hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?|hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?).*?(?:for|during|in|after|within|over|under|about|around|approximately|roughly|exactly|precisely|just|only|at\\s+least|at\\s+most|more\\s+than|less\\s+than|up\\s+to|down\\s+to)\\b",
                "\\b(?:set|setting|start|starting|begin|beginning|create|creating|make|making|put|putting|place|placing|configure|configuring|establish|establishing|initiate|initiating|launch|launching|activate|activating|trigger|triggering|run|running|execute|executing).*?(?:timer|alarm|countdown|stopwatch|clock|time|duration|period).*?(?:for|to|at|in|during|within|over|under|about|around|approximately|roughly|exactly|precisely|just|only|at\\s+least|at\\s+most|more\\s+than|less\\s+than|up\\s+to|down\\s+to)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?|hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?|hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?).*?(?:set|setting|start|starting|begin|beginning|create|creating|make|making|put|putting|place|placing|configure|configuring|establish|establishing|initiate|initiating|launch|launching|activate|activating|trigger|triggering|run|running|execute|executing).*?(?:timer|alarm|countdown|stopwatch|clock|time|duration|period)\\b",
                "\\b(?:timer|alarm|countdown|stopwatch|clock|time|duration|period).*?(?:set|setting|start|starting|begin|beginning|create|creating|make|making|put|putting|place|placing|configure|configuring|establish|establishing|initiate|initiating|launch|launching|activate|activating|trigger|triggering|run|running|execute|executing).*?(?:for|to|at|in|during|within|over|under|about|around|approximately|roughly|exactly|precisely|just|only|at\\s+least|at\\s+most|more\\s+than|less\\s+than|up\\s+to|down\\s+to)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?|hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?|hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?).*?(?:timer|alarm|countdown|stopwatch|clock|time|duration|period).*?(?:set|setting|start|starting|begin|beginning|create|creating|make|making|put|putting|place|placing|configure|configuring|establish|establishing|initiate|initiating|launch|launching|activate|activating|trigger|triggering|run|running|execute|executing)\\b",
                "\\b(?:wake\\s+me|alarm\\s+me|alert\\s+me|remind\\s+me|notify\\s+me|call\\s+me|ring\\s+me|buzz\\s+me|ping\\s+me|sound\\s+off|go\\s+off).*?(?:in|after|within|at|for|during)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?|hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?|hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?).*?(?:wake\\s+me|alarm\\s+me|alert\\s+me|remind\\s+me|notify\\s+me|call\\s+me|ring\\s+me|buzz\\s+me|ping\\s+me|sound\\s+off|go\\s+off)\\b",
                "\\b(?:wait|pause|hold|stop|delay|postpone|suspend|freeze|halt).*?(?:for|during|in|within|over|about|around|approximately|roughly|exactly|precisely|just|only|at\\s+least|at\\s+most|more\\s+than|less\\s+than|up\\s+to|down\\s+to)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?|hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?)\\b",
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?|hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?|seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?).*?(?:wait|pause|hold|stop|delay|postpone|suspend|freeze|halt)\\b",
                // NEW: Direct time patterns for alarm setting
                "\\b(?:set|create|make).*?(?:alarm|wake).*?(?:for|at)\\s*(\\d{3,4})(?:\\s*(?:am|pm))?\\b",
                "\\b(?:set|create|make).*?(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}(?::\\d{2})?)(?:\\s*(?:am|pm))?\\b",
                "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{3,4})(?:\\s*(?:am|pm))?\\b",
                "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}(?::\\d{2})?)(?:\\s*(?:am|pm))?\\b"
            ]
            
            for pattern in timePatterns {
                if let range = text.range(of: pattern, options: .regularExpression) {
                    let matchedText = String(text[range])
                    
                    // Check for time format (like 3:45 or 3:45pm)
                    if let timeMatch = matchedText.range(of: "\\d{1,2}(?::\\d{2})?(?:\\s*[ap]m)?", options: .regularExpression) {
                        let timeString = String(matchedText[timeMatch])
                        // Normalize the time format to 24-hour HH:MM format
                        return normalizeTimeFormat(timeString)
                    }
                    
                    // Check for simple numeric times like "730", "1030" etc. in alarm context
                    if text.range(of: "\\b(?:alarm|wake)", options: .regularExpression) != nil {
                        if let simpleTimeMatch = matchedText.range(of: "\\b\\d{3,4}\\b", options: .regularExpression) {
                            let timeString = String(matchedText[simpleTimeMatch])
                            // If it's in alarm context and looks like a time (3-4 digits), normalize it
                            if timeString.count >= 3 {
                                return normalizeTimeFormat(timeString)
                            }
                        }
                    }
                    
                    // Extract number for duration
                    let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                    if let match = matches.first {
                        let numberRange = Range(match.range, in: matchedText)!
                        let numberString = String(matchedText[numberRange])
                        if let number = Double(numberString) {
                            return Int(number)
                        }
                    }
                }
            }
            
            // Fallback to any number for timers
            let matches = numberRegex.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))
            if let match = matches.first {
                let numberRange = Range(match.range, in: text)!
                let numberString = String(text[numberRange])
                if let number = Double(numberString) {
                    return Int(number)
                }
            }
            
        default:
            // General number extraction for other intents
            let generalPatterns = [
                "\\b(\\d+(?:\\.\\d+)?)\\s*(?:units?|items?|pieces?|counts?|amounts?|quantities?|numbers?|values?|figures?|measurements?|readings?|totals?|sums?)\\b",
                "\\b(?:value|number|amount|quantity|count|total|sum|figure|measurement|reading|score|rating|level|degree|percentage|percent|rate|ratio|proportion).*?(\\d+(?:\\.\\d+)?)\\b",
                "\\b(\\d+(?:\\.\\d+)?).*?(?:value|number|amount|quantity|count|total|sum|figure|measurement|reading|score|rating|level|degree|percentage|percent|rate|ratio|proportion)\\b"
            ]
            
            for pattern in generalPatterns {
                if let range = text.range(of: pattern, options: .regularExpression) {
                    let matchedText = String(text[range])
                    let matches = try! NSRegularExpression(pattern: "(\\d+(?:\\.\\d+)?)", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                    if let match = matches.first {
                        let numberRange = Range(match.range, in: matchedText)!
                        let numberString = String(matchedText[numberRange])
                        if let number = Double(numberString) {
                            return Int(number)
                        }
                    }
                }
            }
            
            // Ultimate fallback - any number
            let matches = numberRegex.matches(in: text, options: [], range: NSRange(location: 0, length: text.count))
            if let match = matches.first {
                let numberRange = Range(match.range, in: text)!
                let numberString = String(text[numberRange])
                if let number = Double(numberString) {
                    return Int(number)
                }
            }
        }
        
        return nil
    }
    
    
    private func extractFeature(text: String) -> String? {
        let features: [String: String] = [
            "do not disturb": "\\b(?:do\\s+not\\s+disturb|dnd|silent\\s+mode|quiet\\s+mode|silence|silent|mute|muted|muting|no\\s+disturb|don't\\s+disturb|dont\\s+disturb|undisturbed|peaceful\\s+mode|focus\\s+mode|concentration\\s+mode)\\b",
            "AOD": "\\b(?:AOD|always\\s+on\\s+display|always-on|always\\s+on|constant\\s+display|persistent\\s+display|continuous\\s+display|permanent\\s+display|screen\\s+always\\s+on|display\\s+always\\s+on|ambient\\s+display|glance\\s+screen|active\\s+display)\\b",
            "raise to wake": "\\b(?:raise\\s+to\\s+wake|lift\\s+to\\s+wake|tap\\s+to\\s+wake|pick\\s+up\\s+to\\s+wake|motion\\s+to\\s+wake|gesture\\s+to\\s+wake|movement\\s+to\\s+wake|touch\\s+to\\s+wake|auto\\s+wake|smart\\s+wake|intelligent\\s+wake|proximity\\s+wake|sensor\\s+wake|accelerometer\\s+wake)\\b",
            "vibration": "\\b(?:vibration|vibrate|vibrating|vibrates|vibrated|haptic|haptics|haptic\\s+feedback|tactile|tactile\\s+feedback|buzz|buzzing|buzzes|buzzed|rumble|rumbling|rumbles|rumbled|shake|shaking|shakes|shook|tremor|trembling|trembles|trembled|pulse|pulsing|pulses|pulsed|vibe|vibes|vibing|vibed)\\b",
            "brightness": "\\b(?:brightness|bright|brighter|brighten|brightest|screen\\s+brightness|display\\s+brightness|backlight|luminosity|luminance|illumination|light\\s+level|intensity|dim|dimmer|dimming|dimmest|dark|darker|darkening|darkest|contrast|gamma|glow|glowing|radiance|brilliance)\\b",
            "volume": "\\b(?:volume|sound\\s+level|audio\\s+level|sound|audio|loud|louder|loudest|quiet|quieter|quietest|soft|softer|softest|high\\s+volume|low\\s+volume|medium\\s+volume|max\\s+volume|min\\s+volume|mute|muted|muting|unmute|unmuted|unmuting|amplify|amplified|amplifying|boost|boosted|boosting|reduce|reduced|reducing)\\b",
            "bluetooth": "\\b(?:bluetooth|bt|blue\\s+tooth|wireless|pairing|paired|pair|connect|connected|connecting|connection|disconnect|disconnected|disconnecting|link|linked|linking|sync|synced|syncing|bond|bonded|bonding)\\b",
            "wifi": "\\b(?:wifi|wi-fi|wi\\s+fi|wireless|wlan|network|internet|connection|connected|connecting|connect|disconnect|disconnected|disconnecting|hotspot|access\\s+point|ssid|router|modem)\\b",
            "location": "\\b(?:location|gps|positioning|coordinates|navigation|maps|directions|tracking|track|tracked|find\\s+my|location\\s+services|geo|geolocation|latitude|longitude|whereabouts|position|place)\\b",
            "flashlight": "\\b(?:flashlight|torch|flash|light|led|lamp|illumination|bright\\s+light|emergency\\s+light|pocket\\s+light|portable\\s+light|handheld\\s+light)\\b",
            "airplane mode": "\\b(?:airplane\\s+mode|flight\\s+mode|aeroplane\\s+mode|air\\s+mode|offline\\s+mode|radio\\s+off|wireless\\s+off|network\\s+off|cellular\\s+off|mobile\\s+off|disconnect\\s+all|isolation\\s+mode)\\b",
            "mobile data": "\\b(?:mobile\\s+data|cellular\\s+data|data\\s+connection|internet\\s+data|4g|5g|3g|lte|edge|hspa|gprs|roaming|data\\s+roaming|cellular|mobile\\s+network|carrier\\s+data)\\b",
            "auto sync": "\\b(?:auto\\s+sync|automatic\\s+sync|sync|syncing|synchronization|synchronize|synchronizing|auto\\s+backup|automatic\\s+backup|cloud\\s+sync|background\\s+sync|real-time\\s+sync)\\b",
            "screen timeout": "\\b(?:screen\\s+timeout|display\\s+timeout|auto\\s+lock|screen\\s+lock|sleep\\s+timer|idle\\s+timeout|standby\\s+timeout|power\\s+saving|screen\\s+saver|auto\\s+sleep|hibernate)\\b"
        ]
        
        // Try to match features in order of specificity (more specific patterns first)
        let sortedFeatures = features.sorted { $0.value.count > $1.value.count }
        
        for (feature, pattern) in sortedFeatures {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return feature
            }
        }
        
        return nil
    }
    
    
    private func extractState(text: String) -> String? {
        if text.range(of: "\\b(?:turn\\s+on|enable|activate|switch\\s+on|start|power\\s+on|boot\\s+up|fire\\s+up|kick\\s+on|bring\\s+up|wake\\s+up|ramp\\s+up|spin\\s+up|light\\s+up|rev\\s+up|crank\\s+up|gear\\s+up|warm\\s+up|dial\\s+up|jack\\s+up|pump\\s+up|open|opening|opened|engage|engaging|engaged|trigger|triggering|triggered|launch|launching|launched|initiate|initiating|initiated|commence|commencing|commenced)\\b", options: .regularExpression) != nil {
            return "on"
        }
        
        if text.range(of: "\\b(?:turn\\s+off|disable|deactivate|switch\\s+off|stop|power\\s+off|shut\\s+down|close\\s+down|wind\\s+down|dial\\s+down|turn\\s+down|shut\\s+off|cut\\s+off|kill|killing|killed|end|ending|ended|finish|finishing|finished|terminate|terminating|terminated|halt|halting|halted|cease|ceasing|ceased|suspend|suspending|suspended|pause|pausing|paused|disengage|disengaging|disengaged|close|closing|closed)\\b", options: .regularExpression) != nil {
            return "off"
        }
        
        if text.range(of: "\\b(?:increase|increasing|increased|up|higher|raise|raising|raised|lift|lifting|lifted|boost|boosting|boosted|amplify|amplifying|amplified|enhance|enhancing|enhanced|escalate|escalating|escalated|intensify|intensifying|intensified|strengthen|strengthening|strengthened|maximize|maximizing|maximized|pump\\s+up|crank\\s+up|ramp\\s+up|step\\s+up|gear\\s+up|rev\\s+up|jack\\s+up|bump\\s+up|turn\\s+up|dial\\s+up|scale\\s+up|beef\\s+up)\\b", options: .regularExpression) != nil {
            return "increase"
        }
        
        if text.range(of: "\\b(?:decrease|decreasing|decreased|down|lower|reduce|reducing|reduced|diminish|diminishing|diminished|lessen|lessening|lessened|weaken|weakening|weakened|minimize|minimizing|minimized|scale\\s+down|dial\\s+down|turn\\s+down|bring\\s+down|take\\s+down|cut\\s+down|wind\\s+down|tone\\s+down|calm\\s+down|cool\\s+down|slow\\s+down|back\\s+down|step\\s+down|gear\\s+down|drop|dropping|dropped)\\b", options: .regularExpression) != nil {
            return "decrease"
        }
        
        return nil
    }
    
    private func extractAction(text: String) -> String? {
        let actions: [String: String] = [
            "call": "\\b(?:call|calling|called|phone|phoning|phoned|dial|dialing|dialed|ring|ringing|rang|contact|contacting|contacted|reach|reaching|reached|get\\s+in\\s+touch|getting\\s+in\\s+touch|got\\s+in\\s+touch|connect\\s+with|connecting\\s+with|connected\\s+with|speak\\s+to|speaking\\s+to|spoke\\s+to|talk\\s+to|talking\\s+to|talked\\s+to)\\b",
            "message": "\\b(?:message|messaging|messaged|text|texting|texted|sms|send\\s+sms|sending\\s+sms|sent\\s+sms|chat|chatting|chatted|write|writing|wrote|compose|composing|composed|draft|drafting|drafted|type|typing|typed|ping|pinging|pinged|notify|notifying|notified|alert|alerting|alerted|inform|informing|informed|tell|telling|told)\\b",
            "video call": "\\b(?:video\\s+call|video\\s+calling|video\\s+called|facetime|face\\s+time|video\\s+chat|video\\s+chatting|video\\s+chatted|skype|skyping|skyped|zoom|zooming|zoomed|hangout|hangouts|meet|meeting|met|conference\\s+call|video\\s+conference|visual\\s+call|cam\\s+call)\\b",
            "set": "\\b(?:set|setting|configure|configuring|configured|setup|setting\\s+up|set\\s+up|establish|establishing|established|create|creating|created|make|making|made|build|building|built|construct|constructing|constructed|form|forming|formed|prepare|preparing|prepared|arrange|arranging|arranged|organize|organizing|organized|define|defining|defined|specify|specifying|specified|determine|determining|determined|fix|fixing|fixed|adjust|adjusting|adjusted|modify|modifying|modified|change|changing|changed|alter|altering|altered|update|updating|updated|edit|editing|edited)\\b",
            "start": "\\b(?:start|starting|started|begin|beginning|began|initiate|initiating|initiated|launch|launching|launched|commence|commencing|commenced|kick\\s+off|kicking\\s+off|kicked\\s+off|fire\\s+up|firing\\s+up|fired\\s+up|boot\\s+up|booting\\s+up|booted\\s+up|power\\s+up|powering\\s+up|powered\\s+up|switch\\s+on|switching\\s+on|switched\\s+on|turn\\s+on|turning\\s+on|turned\\s+on|activate|activating|activated|enable|enabling|enabled|engage|engaging|engaged|trigger|triggering|triggered|open|opening|opened)\\b",
            "stop": "\\b(?:stop|stopping|stopped|end|ending|ended|finish|finishing|finished|complete|completing|completed|conclude|concluding|concluded|terminate|terminating|terminated|halt|halting|halted|cease|ceasing|ceased|quit|quitting|quit|exit|exiting|exited|close|closing|closed|shut\\s+down|shutting\\s+down|shut\\s+down|power\\s+off|powering\\s+off|powered\\s+off|switch\\s+off|switching\\s+off|switched\\s+off|turn\\s+off|turning\\s+off|turned\\s+off|deactivate|deactivating|deactivated|disable|disabling|disabled|suspend|suspending|suspended|pause|pausing|paused|cancel|canceling|canceled|abort|aborting|aborted)\\b",
            "open": "\\b(?:open|opening|opened|launch|launching|launched|start\\s+up|starting\\s+up|started\\s+up|fire\\s+up|firing\\s+up|fired\\s+up|boot\\s+up|booting\\s+up|booted\\s+up|load|loading|loaded|run|running|ran|execute|executing|executed|activate|activating|activated|access|accessing|accessed|enter|entering|entered|go\\s+to|going\\s+to|went\\s+to|navigate\\s+to|navigating\\s+to|navigated\\s+to|visit|visiting|visited|browse|browsing|browsed|view|viewing|viewed|display|displaying|displayed|show|showing|showed|present|presenting|presented)\\b",
            "check": "\\b(?:check|checking|checked|verify|verifying|verified|examine|examining|examined|inspect|inspecting|inspected|review|reviewing|reviewed|look|looking|looked|see|seeing|saw|view|viewing|viewed|observe|observing|observed|monitor|monitoring|monitored|watch|watching|watched|survey|surveying|surveyed|scan|scanning|scanned|browse|browsing|browsed|search|searching|searched|find|finding|found|locate|locating|located|discover|discovering|discovered|explore|exploring|explored|investigate|investigating|investigated|study|studying|studied|analyze|analyzing|analyzed|assess|assessing|assessed|evaluate|evaluating|evaluated)\\b",
            "measure": "\\b(?:measure|measuring|measured|test|testing|tested|record|recording|recorded|log|logging|logged|track|tracking|tracked|monitor|monitoring|monitored|gauge|gauging|gauged|assess|assessing|assessed|evaluate|evaluating|evaluated|calculate|calculating|calculated|compute|computing|computed|determine|determining|determined|quantify|quantifying|quantified|estimate|estimating|estimated|approximate|approximating|approximated|count|counting|counted|tally|tallying|tallied|sum|summing|summed|total|totaling|totaled)\\b",
            "play": "\\b(?:play|playing|played|start\\s+playing|starting\\s+playing|started\\s+playing|begin\\s+playing|beginning\\s+playing|began\\s+playing|resume|resuming|resumed|continue|continuing|continued|run|running|ran|execute|executing|executed|stream|streaming|streamed|broadcast|broadcasting|broadcasted|transmit|transmitting|transmitted)\\b",
            "pause": "\\b(?:pause|pausing|paused|hold|holding|held|suspend|suspending|suspended|freeze|freezing|froze|halt|halting|halted|stop\\s+temporarily|stopping\\s+temporarily|stopped\\s+temporarily|break|breaking|broke|interrupt|interrupting|interrupted|delay|delaying|delayed|wait|waiting|waited)\\b",
            "skip": "\\b(?:skip|skipping|skipped|next|forward|forwarding|forwarded|advance|advancing|advanced|jump\\s+ahead|jumping\\s+ahead|jumped\\s+ahead|move\\s+forward|moving\\s+forward|moved\\s+forward|go\\s+forward|going\\s+forward|went\\s+forward|fast\\s+forward|fast\\s+forwarding|fast\\s+forwarded)\\b",
            "previous": "\\b(?:previous|back|backward|rewinding|rewound|rewind|reverse|reversing|reversed|go\\s+back|going\\s+back|went\\s+back|move\\s+back|moving\\s+back|moved\\s+back|step\\s+back|stepping\\s+back|stepped\\s+back|return|returning|returned|retreat|retreating|retreated)\\b",
            "shuffle": "\\b(?:shuffle|shuffling|shuffled|randomize|randomizing|randomized|mix|mixing|mixed|jumble|jumbling|jumbled|scramble|scrambling|scrambled|random\\s+play|randomly\\s+play|random\\s+order|mixed\\s+order)\\b",
            "repeat": "\\b(?:repeat|repeating|repeated|loop|looping|looped|replay|replaying|replayed|again|once\\s+more|one\\s+more\\s+time|over\\s+again|all\\s+over|from\\s+beginning|from\\s+start|restart|restarting|restarted|redo|redoing|redid|reiterate|reiterating|reiterated)\\b",
            "mute": "\\b(?:mute|muting|muted|silence|silencing|silenced|quiet|quieting|quieted|hush|hushing|hushed|turn\\s+off\\s+sound|turning\\s+off\\s+sound|turned\\s+off\\s+sound|disable\\s+sound|disabling\\s+sound|disabled\\s+sound|cut\\s+sound|cutting\\s+sound|cut\\s+sound|kill\\s+sound|killing\\s+sound|killed\\s+sound)\\b",
            "unmute": "\\b(?:unmute|unmuting|unmuted|enable\\s+sound|enabling\\s+sound|enabled\\s+sound|turn\\s+on\\s+sound|turning\\s+on\\s+sound|turned\\s+on\\s+sound|restore\\s+sound|restoring\\s+sound|restored\\s+sound|bring\\s+back\\s+sound|bringing\\s+back\\s+sound|brought\\s+back\\s+sound)\\b"
        ]
        
        // Sort actions by pattern length for more specific matching
        let sortedActions = actions.sorted { $0.value.count > $1.value.count }
        
        for (action, pattern) in sortedActions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        
        return nil
    }
    
    
    private func extractTimerAction(text: String) -> String? {
        let timerActions: [String: String] = [
            "set": "\\b(?:set|setting|create|creating|make|making|establish|establishing|configure|configuring|start|starting|begin|beginning|initiate|initiating)\\b",
            "start": "\\b(?:start|starting|begin|beginning|run|running|activate|activating|turn\\s+on|turning\\s+on|launch|launching|kick\\s+off|kicking\\s+off)\\b",
            "stop": "\\b(?:stop|stopping|end|ending|finish|finishing|cancel|canceling|halt|halting|terminate|terminating|turn\\s+off|turning\\s+off|disable|disabling)\\b",
            "pause": "\\b(?:pause|pausing|hold|holding|suspend|suspending|freeze|freezing|break|breaking)\\b",
            "resume": "\\b(?:resume|resuming|continue|continuing|restart|restarting|unpause|unpausing)\\b"
        ]
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        let sortedActions = timerActions.sorted { $0.value.count > $1.value.count }
        
        for (action, pattern) in sortedActions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        
        return nil
    }
    
    private func extractMediaAction(text: String) -> String? {
        let mediaActions: [String: String] = [
            "play": "\\b(?:play|playing|played|start\\s+playing|starting\\s+playing|started\\s+playing|begin\\s+playing|beginning\\s+playing|began\\s+playing|resume\\s+playing|resuming\\s+playing|resumed\\s+playing|continue\\s+playing|continuing\\s+playing|continued\\s+playing|run|running|ran|execute|executing|executed|stream|streaming|streamed|broadcast|broadcasting|broadcasted|transmit|transmitting|transmitted|put\\s+on|putting\\s+on|turn\\s+on|turning\\s+on|turned\\s+on|switch\\s+on|switching\\s+on|switched\\s+on|fire\\s+up|firing\\s+up|fired\\s+up|boot\\s+up|booting\\s+up|booted\\s+up|launch|launching|launched|activate|activating|activated|start|starting|started)\\b",
            "pause": "\\b(?:pause|pausing|paused|hold|holding|held|suspend|suspending|suspended|freeze|freezing|froze|halt|halting|halted|stop\\s+temporarily|stopping\\s+temporarily|stopped\\s+temporarily|break|breaking|broke|interrupt|interrupting|interrupted|delay|delaying|delayed|wait|waiting|waited|rest|resting|rested)\\b",
            "stop": "\\b(?:stop|stopping|stopped|end|ending|ended|finish|finishing|finished|complete|completing|completed|conclude|concluding|concluded|terminate|terminating|terminated|halt|halting|halted|cease|ceasing|ceased|quit|quitting|exit|exiting|exited|close|closing|closed|shut\\s+down|shutting\\s+down|shut\\s+down|power\\s+off|powering\\s+off|powered\\s+off|switch\\s+off|switching\\s+off|switched\\s+off|turn\\s+off|turning\\s+off|turned\\s+off|cut\\s+off|cutting\\s+off|cut\\s+off|kill|killing|killed)\\b",
            "skip": "\\b(?:skip|skipping|skipped|next|forward|forwarding|forwarded|advance|advancing|advanced|jump\\s+ahead|jumping\\s+ahead|jumped\\s+ahead|move\\s+forward|moving\\s+forward|moved\\s+forward|go\\s+forward|going\\s+forward|went\\s+forward|fast\\s+forward|fast\\s+forwarding|fast\\s+forwarded|skip\\s+ahead|skipping\\s+ahead|skipped\\s+ahead|jump\\s+forward|jumping\\s+forward|jumped\\s+forward|leap\\s+ahead|leaping\\s+ahead|leaped\\s+ahead|step\\s+forward|stepping\\s+forward|stepped\\s+forward)\\b",
            "previous": "\\b(?:previous|back|backward|rewinding|rewound|rewind|reverse|reversing|reversed|go\\s+back|going\\s+back|went\\s+back|move\\s+back|moving\\s+back|moved\\s+back|step\\s+back|stepping\\s+back|stepped\\s+back|jump\\s+back|jumping\\s+back|jumped\\s+back|skip\\s+back|skipping\\s+back|skipped\\s+back|return|returning|returned|retreat|retreating|retreated|backtrack|backtracking|backtracked|reverse|reversing|reversed|last|prior|earlier|before)\\b",
            "shuffle": "\\b(?:shuffle|shuffling|shuffled|randomize|randomizing|randomized|mix|mixing|mixed|jumble|jumbling|jumbled|scramble|scrambling|scrambled|random\\s+play|randomly\\s+play|random\\s+order|mixed\\s+order|disorder|disordering|disordered|mess\\s+up|messing\\s+up|messed\\s+up|rearrange|rearranging|rearranged)\\b",
            "repeat": "\\b(?:repeat|repeating|repeated|loop|looping|looped|replay|replaying|replayed|again|once\\s+more|one\\s+more\\s+time|over\\s+again|all\\s+over|from\\s+beginning|from\\s+start|restart|restarting|restarted|redo|redoing|redid|reiterate|reiterating|reiterated|cycle|cycling|cycled|continuous|continuously|endlessly|infinitely)\\b",
            "mute": "\\b(?:mute|muting|muted|silence|silencing|silenced|quiet|quieting|quieted|hush|hushing|hushed|shush|shushing|shushed|turn\\s+off\\s+sound|turning\\s+off\\s+sound|turned\\s+off\\s+sound|disable\\s+sound|disabling\\s+sound|disabled\\s+sound|cut\\s+sound|cutting\\s+sound|cut\\s+sound|kill\\s+sound|killing\\s+sound|killed\\s+sound|no\\s+sound|without\\s+sound|soundless|voiceless)\\b",
            "unmute": "\\b(?:unmute|unmuting|unmuted|enable\\s+sound|enabling\\s+sound|enabled\\s+sound|turn\\s+on\\s+sound|turning\\s+on\\s+sound|turned\\s+on\\s+sound|restore\\s+sound|restoring\\s+sound|restored\\s+sound|bring\\s+back\\s+sound|bringing\\s+back\\s+sound|brought\\s+back\\s+sound|activate\\s+sound|activating\\s+sound|activated\\s+sound|switch\\s+on\\s+sound|switching\\s+on\\s+sound|switched\\s+on\\s+sound|unsilence|unsilencing|unsilenced)\\b",
            "volume up": "\\b(?:volume\\s+up|turn\\s+up|turning\\s+up|turned\\s+up|increase\\s+volume|increasing\\s+volume|increased\\s+volume|raise\\s+volume|raising\\s+volume|raised\\s+volume|boost\\s+volume|boosting\\s+volume|boosted\\s+volume|amplify|amplifying|amplified|louder|make\\s+louder|making\\s+louder|made\\s+louder|crank\\s+up|cranking\\s+up|cranked\\s+up|pump\\s+up|pumping\\s+up|pumped\\s+up)\\b",
            "volume down": "\\b(?:volume\\s+down|turn\\s+down|turning\\s+down|turned\\s+down|decrease\\s+volume|decreasing\\s+volume|decreased\\s+volume|lower\\s+volume|lowering\\s+volume|lowered\\s+volume|reduce\\s+volume|reducing\\s+volume|reduced\\s+volume|quieter|make\\s+quieter|making\\s+quieter|made\\s+quieter|softer|make\\s+softer|making\\s+softer|made\\s+softer)\\b"
        ]
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        let sortedActions = mediaActions.sorted { $0.value.count > $1.value.count }
        
        for (action, pattern) in sortedActions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        
        return nil
    }
    
    private func extractAppAction(text: String) -> String? {
        let appActions: [String: String] = [
            "open": "\\b(?:open|opening|opened|launch|launching|launched|start|starting|started|run|running|ran|execute|executing|executed|activate|activating|activated|fire\\s+up|firing\\s+up|fired\\s+up|boot\\s+up|booting\\s+up|booted\\s+up|load|loading|loaded)\\b",
            "close": "\\b(?:close|closing|closed|shut|shutting|shut|exit|exiting|exited|quit|quitting|terminate|terminating|terminated|end|ending|ended|finish|finishing|finished|stop|stopping|stopped)\\b",
            "switch": "\\b(?:switch|switching|switched|change|changing|changed|go\\s+to|going\\s+to|went\\s+to|navigate\\s+to|navigating\\s+to|navigated\\s+to|move\\s+to|moving\\s+to|moved\\s+to|jump\\s+to|jumping\\s+to|jumped\\s+to)\\b"
        ]
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        let sortedActions = appActions.sorted { $0.value.count > $1.value.count }
        
        for (action, pattern) in sortedActions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        
        return nil
    }
    
    private func extractPhoneAction(text: String) -> String? {
        let phoneActions: [String: String] = [
            "call": "\\b(?:call|calling|called|phone|phoning|phoned|dial|dialing|dialed|ring|ringing|rang)\\b",
            "text": "\\b(?:text|texting|texted|message|messaging|messaged|sms|send\\s+sms|sending\\s+sms|sent\\s+sms)\\b"
        ]
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        let sortedActions = phoneActions.sorted { $0.value.count > $1.value.count }
        
        for (action, pattern) in sortedActions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        
        return nil
    }
    
    private func extractTool(text: String) -> String? {
        if text.range(of: "\\b(?:alarm|alarms|wake\\s+up|wake\\s+me|wake\\s+me\\s+up|morning\\s+alarm|daily\\s+alarm|recurring\\s+alarm|set\\s+alarm|setting\\s+alarm|set\\s+alarm|alarm\\s+clock|clock\\s+alarm|wakeup\\s+call|wake\\s+up\\s+call|reveille)\\b", options: .regularExpression) != nil {
            return "alarm"
        }
        
        if text.range(of: "\\b(?:timer|timers|countdown|count\\s+down|counting\\s+down|kitchen\\s+timer|cooking\\s+timer|egg\\s+timer|stopwatch\\s+timer|interval\\s+timer|pomodoro|pomo|break\\s+timer|work\\s+timer|study\\s+timer|meditation\\s+timer|exercise\\s+timer|reminder\\s+timer)\\b", options: .regularExpression) != nil {
            return "timer"
        }
        
        if text.range(of: "\\b(?:stopwatch|stop\\s+watch|chronometer|chrono|lap\\s+timer|split\\s+timer|race\\s+timer|sports\\s+timer|athletic\\s+timer|precision\\s+timer|accurate\\s+timer|timing\\s+device|time\\s+keeper|timekeeper)\\b", options: .regularExpression) != nil {
            return "stopwatch"
        }
        
        return nil
    }
    
    
    private func extractActivityType(text: String) -> String? {
        let activities: [String: String] = [
            "outdoor run": "\\b(?:outdoor\\s+)?(?:run|running|jog|jogging|sprint|sprinting|marathon|half\\s+marathon|5k|10k|race|racing|trail\\s+run|trail\\s+running|road\\s+run|road\\s+running|distance\\s+run|distance\\s+running|long\\s+run|long\\s+distance\\s+run|endurance\\s+run|cardio\\s+run)\\b",
            "indoor cycling": "\\b(?:indoor\\s+)?(?:cycling|cycle|bike|biking|bicycle|bicycling|spin|spinning|stationary\\s+bike|exercise\\s+bike|indoor\\s+bike|gym\\s+bike|fitness\\s+bike|spin\\s+class|cycling\\s+class|bike\\s+workout|cycling\\s+workout|pedal|pedaling)\\b",
            "swimming": "\\b(?:swim|swimming|swam|pool|pools|lap|laps|freestyle|backstroke|breaststroke|butterfly|water\\s+aerobics|aqua\\s+fitness|pool\\s+workout|water\\s+workout|diving|synchronized\\s+swimming|open\\s+water\\s+swimming|ocean\\s+swimming|lake\\s+swimming)\\b",
            "yoga": "\\b(?:yoga|yogi|asana|asanas|namaste|meditation|meditate|meditated|stretch|stretching|stretched|flexibility|mindfulness|breathing\\s+exercise|pranayama|vinyasa|hatha|ashtanga|bikram|hot\\s+yoga|power\\s+yoga|restorative\\s+yoga|yin\\s+yoga|kundalini)\\b",
            "walking": "\\b(?:walk|walking|walked|stroll|strolling|strolled|hike|hiking|hiked|trek|trekking|trekked|march|marching|marched|pace|pacing|paced|step|stepping|stepped|footstep|footsteps|stride|striding|strode|amble|ambling|ambled|saunter|sauntering|sauntered|wander|wandering|wandered)\\b",
            "workout": "\\b(?:workout|work\\s+out|working\\s+out|worked\\s+out|exercise|exercising|exercised|training|train|trained|gym|gymnasium|fitness|fit|strength\\s+training|weight\\s+training|resistance\\s+training|cardio|cardiovascular|aerobic|anaerobic|circuit\\s+training|interval\\s+training|hiit|high\\s+intensity|crossfit|bodybuilding|powerlifting|weightlifting|calisthenics|bodyweight|functional\\s+training)\\b",
            "tennis": "\\b(?:tennis|ping\\s+pong|table\\s+tennis|badminton|squash|racquetball|paddle\\s+tennis|platform\\s+tennis)\\b",
            "basketball": "\\b(?:basketball|hoops|ball|dribble|dribbling|shoot|shooting|layup|three\\s+pointer|free\\s+throw|dunk|dunking)\\b",
            "football": "\\b(?:football|soccer|kick|kicking|goal|penalty|corner\\s+kick|free\\s+kick)\\b",
            "baseball": "\\b(?:baseball|bat|batting|pitch|pitching|home\\s+run|strike|ball|inning)\\b",
            "golf": "\\b(?:golf|golfing|golfer|drive|driving|putt|putting|chip|chipping|tee\\s+off|fairway|green|bunker|sand\\s+trap|hole\\s+in\\s+one|birdie|eagle|par|bogey)\\b",
            "dancing": "\\b(?:dance|dancing|danced|dancer|ballet|ballroom|salsa|tango|waltz|foxtrot|swing|latin|hip\\s+hop|contemporary|jazz\\s+dance|tap\\s+dance|breakdance|choreography)\\b",
            "martial arts": "\\b(?:martial\\s+arts|karate|judo|taekwondo|kung\\s+fu|boxing|kickboxing|muay\\s+thai|jiu\\s+jitsu|aikido|krav\\s+maga|self\\s+defense|sparring|kata|forms)\\b",
            "climbing": "\\b(?:climb|climbing|climbed|rock\\s+climb|rock\\s+climbing|bouldering|mountaineering|rappelling|belaying)\\b",
            "skiing": "\\b(?:ski|skiing|skied|snowboard|snowboarding|snowboarded|alpine|nordic|cross\\s+country|downhill|slalom|mogul|freestyle)\\b"
        ]
        
        // Sort by pattern length for more specific matching
        let sortedActivities = activities.sorted { $0.value.count > $1.value.count }
        
        for (activity, pattern) in sortedActivities {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return activity
            }
        }
        
        return nil
    }
    
    private func extractApp(text: String) -> String? {
        let apps: [String: String] = [
            "weather": "\\b(?:weather|forecast|temperature|temp|rain|snow|sunny|cloudy|humidity|wind|storm|thunder|lightning|precipitation|meteorology|climate|conditions|barometric|pressure|uv\\s+index|heat\\s+index|wind\\s+chill|dew\\s+point|visibility|sunrise|sunset|moonrise|moonset)\\b",
            "settings": "\\b(?:settings?|preferences|prefs|config|configuration|options|controls|setup|system|admin|administration|customize|customization|personalize|personalization)\\b",
            "health": "\\b(?:health|fitness|medical|wellness|wellbeing|doctor|physician|hospital|clinic|symptoms|diagnosis|treatment|medicine|medication|prescription|vital\\s+signs|blood\\s+pressure|heart\\s+rate|temperature|pulse|medical\\s+record|health\\s+record|fitness\\s+tracker|activity\\s+tracker)\\b",
            "calendar": "\\b(?:calendar|schedule|appointment|meeting|event|date|time|reminder|alert|notification|agenda|planner|diary|booking|reservation|commitment|engagement|conference|session)\\b",
            "camera": "\\b(?:camera|photo|picture|image|snapshot|selfie|portrait|landscape|video|record|recording|shoot|shooting|capture|capturing|lens|flash|zoom|focus|exposure|shutter)\\b",
            "music": "\\b(?:music|song|songs|track|tracks|album|albums|artist|artists|band|bands|playlist|playlists|audio|sound|melody|rhythm|beat|tune|harmony|compose|composition|lyrics|genre|spotify|apple\\s+music|pandora|youtube\\s+music|soundcloud)\\b",
            "maps": "\\b(?:maps?|navigation|navigate|directions|route|routing|gps|location|address|destination|waypoint|landmark|street|road|highway|freeway|avenue|boulevard|intersection|coordinates|latitude|longitude|compass|north|south|east|west|traffic|commute)\\b",
            "messages": "\\b(?:messages?|messaging|text|texting|sms|chat|chatting|conversation|thread|imessage|whatsapp|telegram|signal|messenger|facetime|video\\s+call|voice\\s+call|communication|correspond|correspondence)\\b",
            "email": "\\b(?:email|e-mail|mail|inbox|outbox|sent|draft|compose|reply|forward|attachment|subject|recipient|sender|cc|bcc|gmail|outlook|yahoo\\s+mail|icloud\\s+mail|exchange|smtp|pop|imap)\\b",
            "photos": "\\b(?:photos?|pictures?|images?|gallery|album|albums|slideshow|memories|moments|library|collection|thumbnail|preview|edit|editing|filter|filters|crop|cropping|rotate|rotation|brightness|contrast|saturation|exposure)\\b",
            "notes": "\\b(?:notes?|note|notepad|memo|memos|reminder|reminders|list|lists|todo|to-do|task|tasks|checklist|journal|diary|text\\s+editor|document|documents|writing|write|typed|type)\\b",
            "calculator": "\\b(?:calculator|calculate|calculation|math|mathematics|arithmetic|add|addition|subtract|subtraction|multiply|multiplication|divide|division|equation|formula|compute|computation|numbers?|digits?)\\b",
            "clock": "\\b(?:clock|time|timer|alarm|stopwatch|world\\s+clock|timezone|time\\s+zone|hour|minute|second|am|pm|24\\s+hour|12\\s+hour|analog|digital)\\b",
            "browser": "\\b(?:browser|web|internet|website|url|link|search|google|safari|chrome|firefox|edge|bookmark|history|tab|tabs|page|pages|browse|browsing|online|www|http|https)\\b"
        ]
        
        // Sort by pattern length for more specific matching
        let sortedApps = apps.sorted { $0.value.count > $1.value.count }
        
        for (app, pattern) in sortedApps {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return app
            }
        }
        
        return nil
    }
    
    
    private func extractContact(text: String) -> String? {
        // First check for hardcoded contacts
        let contacts: [String: String] = [
            "mom": "\\b(?:mom|mother|mama|mum|mummy|ma|mommy|mamma|momma)\\b",
            "dad": "\\b(?:dad|father|papa|pop|daddy|pappa|pops|pa|old\\s+man)\\b",
            "sister": "\\b(?:sister|sis|sibling)\\b",
            "brother": "\\b(?:brother|bro|sibling)\\b"
        ]
        
        for (contact, pattern) in contacts {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return contact
            }
        }
        
        // Special case: check for emergency services first
        let emergencyServices: [String: String] = [
            "911": "\\b(?:911|emergency|police|fire\\s+department|ambulance|paramedics)\\b",
            "emergency": "\\b(?:emergency|urgent|help|sos|crisis|disaster|accident|medical\\s+emergency)\\b"
        ]
        
        for (service, pattern) in emergencyServices {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return service
            }
        }
        
        // If no hardcoded contact found, try to extract any name after phone action keywords
        let phoneActionPatterns = [
            "\\b(?:call|calling|called|phone|phoning|phoned|dial|dialing|dialed|ring|ringing|rang|contact|contacting|contacted|reach|reaching|reached|get\\s+in\\s+touch\\s+with|getting\\s+in\\s+touch\\s+with|got\\s+in\\s+touch\\s+with)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "\\b(?:text|texting|texted|message|messaging|messaged|sms|send\\s+sms\\s+to|sending\\s+sms\\s+to|sent\\s+sms\\s+to|send\\s+message\\s+to|sending\\s+message\\s+to|sent\\s+message\\s+to|write\\s+to|writing\\s+to|wrote\\s+to)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "\\b(?:video\\s+call|video\\s+calling|video\\s+called|facetime|face\\s+time|skype|skyping|skyped|zoom|zooming|zoomed)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "([A-Za-z][A-Za-z\\s]{1,30})\\s+(?:please|pls)",
            "(?:hey|hi|hello|yo)\\s+([A-Za-z][A-Za-z\\s]{1,30})",
            "\\b(?:to|for)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "([A-Za-z][A-Za-z\\s]{1,30})(?:'s|'s)\\s+(?:number|phone|mobile|cell)",
            "\\b(?:contact|person|friend|family|colleague|neighbor|relative)\\s+(?:named|called)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "([A-Za-z][A-Za-z\\s]{1,30})\\s+(?:from|at)\\s+(?:work|office|home|school|gym|church|club)",
            "\\b(?:dr|doctor|mr|mister|mrs|miss|ms)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)",
            "([A-Za-z][A-Za-z\\s]{1,30})\\s+(?:the|my)\\s+(?:doctor|dentist|lawyer|teacher|boss|manager|supervisor|assistant|secretary|colleague|friend|neighbor|relative|family|cousin|aunt|uncle|grandmother|grandfather|grandma|grandpa|nephew|niece)",
            "\\b(?:my|the)\\s+(?:doctor|dentist|lawyer|teacher|boss|manager|supervisor|assistant|secretary|colleague|friend|neighbor|relative|family|cousin|aunt|uncle|grandmother|grandfather|grandma|grandpa|nephew|niece)\\s+([A-Za-z][A-Za-z\\s]{1,30})(?:\\s|$|\\.|\\?|!)"
        ]
        
        for pattern in phoneActionPatterns {
            if let range = text.range(of: pattern, options: .regularExpression) {
                let matchedText = String(text[range])
                let matches = try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive]).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                
                if let match = matches.first, match.numberOfRanges > 1 {
                    let nameRange = Range(match.range(at: 1), in: matchedText)!
                    let name = String(matchedText[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                    
                    // Validate extracted name
                    if !name.isEmpty && 
                       name.count >= 2 && 
                       name.count <= 30 &&
                       !["the", "my", "a", "an", "to", "for", "at", "in", "on", "with", "by", "from", "of", "and", "or", "but", "please", "pls", "hey", "hi", "hello", "yo", "now", "today", "tomorrow", "yesterday", "later", "soon", "quickly", "slowly", "again", "back", "home", "work", "office", "number", "phone", "mobile", "cell", "contact", "person", "friend", "family", "someone", "anybody", "anyone", "somebody", "nobody", "everyone", "everybody"].contains(name.lowercased()) {
                        return name
                    }
                }
            }
        }
        
        // Additional pattern specifically for phone numbers with common formats
        let phoneNumberPatterns = [
            "\\b(\\d{3}[-.]?\\d{3}[-.]?\\d{4})\\b",
            "\\b(\\(\\d{3}\\)\\s?\\d{3}[-.]?\\d{4})\\b",
            "\\b(\\+\\d{1,3}\\s?\\d{3,4}\\s?\\d{3,4}\\s?\\d{3,4})\\b",
            "\\b(\\d{10,15})\\b",
            "\\b(?:number|phone|mobile|cell)\\s+(\\d{3}[-.]?\\d{3}[-.]?\\d{4})\\b",
            "\\b(?:number|phone|mobile|cell)\\s+(\\(\\d{3}\\)\\s?\\d{3}[-.]?\\d{4})\\b",
            "\\b(?:call|dial|phone)\\s+(\\d{3}[-.]?\\d{3}[-.]?\\d{4})\\b",
            "\\b(?:call|dial|phone)\\s+(\\(\\d{3}\\)\\s?\\d{3}[-.]?\\d{4})\\b"
        ]
        
        for pattern in phoneNumberPatterns {
            if let range = text.range(of: pattern, options: .regularExpression) {
                let matchedText = String(text[range])
                let matches = try! NSRegularExpression(pattern: "(\\d{3}[-.]?\\d{3}[-.]?\\d{4}|\\(\\d{3}\\)\\s?\\d{3}[-.]?\\d{4}|\\+\\d{1,3}\\s?\\d{3,4}\\s?\\d{3,4}\\s?\\d{3,4}|\\d{10,15})", options: []).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
                
                if let match = matches.first {
                    let numberRange = Range(match.range, in: matchedText)!
                    let phoneNumber = String(matchedText[numberRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                    
                    if !phoneNumber.isEmpty {
                        return phoneNumber
                    }
                }
            }
        }
        
        return nil
    }
    
    
    private func extractLocation(text: String) -> String? {
        // First check for hardcoded locations
        let locations: [String: String] = [
            "london": "\\b(?:london|uk|england|britain|great\\s+britain|united\\s+kingdom|british)\\b",
            "bangalore": "\\b(?:bangalore|bengaluru|blr|karnataka|india|indian)\\b",
            "mumbai": "\\b(?:mumbai|bombay|maharashtra|india|indian)\\b",
            "delhi": "\\b(?:delhi|new\\s+delhi|ncr|national\\s+capital\\s+region|india|indian)\\b"
        ]
        
        for (location, pattern) in locations {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return location
            }
        }
        
        // Try to extract location after weather-related keywords
        let weatherPattern = "(?:weather|forecast|temperature|climate|conditions)\\s+(.+?)(?:\\s|$|\\.|\\?|!)"
        if let range = text.range(of: weatherPattern, options: .regularExpression) {
            let matchedText = String(text[range])
            let matches = try! NSRegularExpression(pattern: weatherPattern, options: [.caseInsensitive]).matches(in: matchedText, options: [], range: NSRange(location: 0, length: matchedText.count))
            
            if let match = matches.first, match.numberOfRanges > 1 {
                let locationRange = Range(match.range(at: 1), in: matchedText)!
                let extractedLocation = String(matchedText[locationRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                
                if !extractedLocation.isEmpty && 
                   extractedLocation.count >= 2 && 
                   extractedLocation.count <= 50 &&
                   !["in", "at", "for", "the", "my", "current", "here", "there", "today", "tomorrow", "now"].contains(extractedLocation.lowercased()) {
                    return extractedLocation
                }
            }
        }
        
        return "current location"  // Default
    }
    
    private func extractAttribute(text: String) -> String? {
        let attributes: [String: String] = [
            "forecast": "\\b(?:forecast|prediction|outlook|future|tomorrow|next\\s+week|next\\s+few\\s+days|coming\\s+days|upcoming|later|ahead|projected|expected|anticipated)\\b",
            "temperature": "\\b(?:temperature|temp|hot|cold|warm|cool|degree|degrees|celsius|fahrenheit|heat|chill|thermal|thermostat)\\b",
            "rain": "\\b(?:rain|raining|shower|showers|precipitation|drizzle|drizzling|downpour|wet|umbrella|storm|storms|thunderstorm|thunderstorms)\\b",
            "humidity": "\\b(?:humidity|humid|moisture|muggy|sticky|damp|dry|arid|steamy|moist)\\b",
            "air quality": "\\b(?:air\\s+quality|aqi|pollution|smog|clean\\s+air|dirty\\s+air|particulates|pm2\\.5|pm10|ozone|carbon\\s+monoxide|allergens|pollen)\\b",
            "wind": "\\b(?:wind|windy|breeze|breezy|gust|gusty|calm|still|draft|drafty)\\b",
            "visibility": "\\b(?:visibility|clear|cloudy|foggy|hazy|misty|overcast|sunny|bright)\\b",
            "uv index": "\\b(?:uv\\s+index|ultraviolet|sun\\s+exposure|sunburn|sun\\s+protection|spf)\\b"
        ]
        
        for (attr, pattern) in attributes {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return attr
            }
        }
        
        return nil
    }
    
    private func extractType(text: String) -> String? {
        if text.range(of: "\\b(?:above|over|exceed|exceeding|exceeded|exceeds|higher|more\\s+than|greater\\s+than|beyond|surpass|surpassing|surpassed|surpasses|top|topping|topped|tops|beat|beating|beaten|beats|cross|crossing|crossed|crosses|pass|passing|passed|passes)\\b", options: .regularExpression) != nil {
            return "high"
        }
        
        if text.range(of: "\\b(?:below|under|less\\s+than|lower\\s+than|beneath|underneath|drop|drops|dropped|dropping|fall|falls|fell|falling|decrease|decreases|decreased|decreasing|reduce|reduces|reduced|reducing|decline|declines|declined|declining|go\\s+down|goes\\s+down|went\\s+down|going\\s+down|dip|dips|dipped|dipping|plunge|plunges|plunged|plunging|sink|sinks|sank|sinking|tumble|tumbles|tumbled|tumbling)\\b", options: .regularExpression) != nil {
            return "low"
        }
        
        return nil
    }
    
    private func extractPeriod(text: String) -> String? {
        let periods: [String: String] = [
            "daily": "\\b(?:daily|every\\s+day|each\\s+day|day\\s+by\\s+day|per\\s+day|on\\s+a\\s+daily\\s+basis|day\\s+to\\s+day|everyday|each\\s+and\\s+every\\s+day)\\b",
            "weekly": "\\b(?:weekly|every\\s+week|each\\s+week|week\\s+by\\s+week|per\\s+week|on\\s+a\\s+weekly\\s+basis|week\\s+to\\s+week|every\\s+seven\\s+days|once\\s+a\\s+week|once\\s+per\\s+week)\\b",
            "monthly": "\\b(?:monthly|every\\s+month|each\\s+month|month\\s+by\\s+month|per\\s+month|on\\s+a\\s+monthly\\s+basis|month\\s+to\\s+month|once\\s+a\\s+month|once\\s+per\\s+month|every\\s+thirty\\s+days)\\b",
            "yearly": "\\b(?:yearly|annually|every\\s+year|each\\s+year|year\\s+by\\s+year|per\\s+year|on\\s+a\\s+yearly\\s+basis|year\\s+to\\s+year|once\\s+a\\s+year|once\\s+per\\s+year|every\\s+365\\s+days)\\b",
            "hourly": "\\b(?:hourly|every\\s+hour|each\\s+hour|hour\\s+by\\s+hour|per\\s+hour|on\\s+an\\s+hourly\\s+basis|hour\\s+to\\s+hour|once\\s+an\\s+hour|once\\s+per\\s+hour|every\\s+sixty\\s+minutes)\\b"
        ]
        
        // Sort by pattern length for more specific matching
        let sortedPeriods = periods.sorted { $0.value.count > $1.value.count }
        
        for (period, pattern) in sortedPeriods {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return period
            }
        }
        
        return nil
    }
    
    private func extractEventType(text: String) -> String? {
        if text.range(of: "\\b(?:weight|weigh|weighing|weighed|kg|kilogram|kilograms|pounds?|lbs?|lb|body\\s+weight|body\\s+mass|mass|scale|heavy|light|bmi|body\\s+mass\\s+index)\\b", options: .regularExpression) != nil {
            return "weight"
        }
        
        if text.range(of: "\\b(?:menstrual|period|cycle|menstruation|menses|flow|pms|ovulation|ovulating|ovulated|fertility|fertile|reproductive|hormonal|cramps|bloating|spotting|irregular|regular)\\b", options: .regularExpression) != nil {
            return "menstrual cycle"
        }
        
        if text.range(of: "\\b(?:medication|medicine|pill|pills|tablet|tablets|capsule|capsules|dose|dosage|prescription|drug|drugs|supplement|supplements|vitamin|vitamins|antibiotic|antibiotics|painkiller|painkillers)\\b", options: .regularExpression) != nil {
            return "medication"
        }
        
        if text.range(of: "\\b(?:blood\\s+pressure|bp|systolic|diastolic|hypertension|hypotension|mmhg|pressure|cardiovascular|heart\\s+health)\\b", options: .regularExpression) != nil {
            return "blood pressure"
        }
        
        if text.range(of: "\\b(?:blood\\s+sugar|glucose|diabetes|diabetic|insulin|blood\\s+glucose|bg|hyperglycemia|hypoglycemia|a1c|hemoglobin\\s+a1c)\\b", options: .regularExpression) != nil {
            return "blood sugar"
        }
        
        if text.range(of: "\\b(?:water|drink|drinking|drank|hydration|hydrated|hydrating|fluid|fluids|liquid|liquids|h2o|ml|milliliters|millilitres|liters|litres|cups?|glasses?|bottles?)\\b", options: .regularExpression) != nil {
            return "water intake"
        }
        
        if text.range(of: "\\b(?:mood|emotion|emotional|feeling|feelings|happy|sad|angry|depressed|anxious|stressed|calm|peaceful|excited|nervous|worried|content|joyful|frustrated|irritated|overwhelmed|relaxed)\\b", options: .regularExpression) != nil {
            return "mood"
        }
        
        return nil
    }
    
    
    // MARK: - Helper Methods
    
    private func addContextualSlots(text: String, intent: String, slots: inout [String: Any]) {
        switch intent {
        case "QueryPoint":
            // Try to infer missing metric
            if slots["metric"] == nil {
                if let inferredMetric = inferMetricFromContext(text: text) {
                    slots["metric"] = inferredMetric
                }
            }
            
            // Add default time_ref if not present
            if slots["time_ref"] == nil {
                slots["time_ref"] = "today"
            }
            
            // Add default unit based on metric
            if slots["unit"] == nil, let metric = slots["metric"] as? String {
                switch metric {
                case "steps":
                    slots["unit"] = "count"
                case "distance":
                    slots["unit"] = "km"
                case "calories":
                    slots["unit"] = "kcal"
                case "heart rate":
                    slots["unit"] = "bpm"
                case "sleep":
                    slots["unit"] = "hours"
                case "weight":
                    slots["unit"] = "kg"
                case "spo2":
                    slots["unit"] = "percent"
                case "stress":
                    slots["unit"] = "score"
                default:
                    break
                }
            }
            
            // Add default identifier if not present
            if slots["identifier"] == nil {
                slots["identifier"] = "average"
            }
            
        case "SetGoal", "SetThreshold":
            // Add default unit based on metric
            if slots["unit"] == nil, let metric = slots["metric"] as? String {
                switch metric {
                case "steps":
                    slots["unit"] = "count"
                case "distance":
                    slots["unit"] = "km"
                case "calories":
                    slots["unit"] = "kcal"
                case "heart rate":
                    slots["unit"] = "bpm"
                case "sleep":
                    slots["unit"] = "hours"
                case "weight":
                    slots["unit"] = "kg"
                case "spo2":
                    slots["unit"] = "percent"
                case "stress":
                    slots["unit"] = "score"
                default:
                    break
                }
            }
            
        case "TimerStopwatch":
            // For timer/stopwatch, try to extract more specific actions
            if slots["action"] == nil {
                if let timerAction = extractTimerAction(text: text) {
                    slots["action"] = timerAction
                }
            }
            
            // Extract value for timer/alarm setting if not already present
            if slots["value"] == nil {
                if let value = extractValue(text: text, intent: intent) {
                    slots["value"] = value
                }
            }
            
        case "PhoneAction":
            // For phone actions, try to extract more specific actions
            if slots["action"] == nil {
                if let phoneAction = extractPhoneAction(text: text) {
                    slots["action"] = phoneAction
                }
            }
            
        case "MediaAction":
            // For media actions, try to extract more specific actions
            if slots["action"] == nil {
                if let mediaAction = extractMediaAction(text: text) {
                    slots["action"] = mediaAction
                }
            }
            
            // Add default target if action is play
            if slots["target"] == nil, let action = slots["action"] as? String {
                if action == "play" {
                    slots["target"] = "music"
                }
            }
            
        case "OpenApp":
            // For app actions, try to extract more specific actions
            if slots["action"] == nil {
                if let appAction = extractAppAction(text: text) {
                    slots["action"] = appAction
                } else {
                    slots["action"] = "open"  // Default action
                }
            }
            
            // Add default target
            if slots["target"] == nil {
                slots["target"] = "app"
            }
            
        case "LogEvent":
            // Add default value and unit for weight logging
            if slots["value"] == nil && slots["event_type"] as? String == "weight" {
                // Try to extract weight value again with more patterns
                if let value = extractValue(text: text, intent: intent) {
                    slots["value"] = value
                }
            }
            
            // Add default unit based on event type
            if slots["unit"] == nil, let eventType = slots["event_type"] as? String {
                switch eventType {
                case "weight":
                    slots["unit"] = "kg"
                case "water intake":
                    slots["unit"] = "ml"
                case "medication":
                    slots["unit"] = "dose"
                case "blood pressure":
                    slots["unit"] = "mmHg"
                case "blood sugar":
                    slots["unit"] = "mg/dL"
                default:
                    break
                }
            }
            
            // Add default time_ref
            if slots["time_ref"] == nil {
                slots["time_ref"] = "now"
            }
            
        case "WeatherQuery":
            // Add default attribute if not present
            if slots["attribute"] == nil {
                slots["attribute"] = "forecast"
            }
            
        case "QueryTrend":
            // Add default period and unit
            if slots["period"] == nil {
                slots["period"] = "weekly"
            }
            
            if slots["unit"] == nil, let metric = slots["metric"] as? String {
                switch metric {
                case "steps":
                    slots["unit"] = "count"
                case "distance":
                    slots["unit"] = "km"
                case "calories":
                    slots["unit"] = "kcal"
                case "heart rate":
                    slots["unit"] = "bpm"
                case "sleep":
                    slots["unit"] = "hours"
                case "weight":
                    slots["unit"] = "kg"
                case "spo2":
                    slots["unit"] = "percent"
                case "stress":
                    slots["unit"] = "score"
                default:
                    break
                }
            }
            
        default:
            break
        }
    }
    
    private func inferMetricFromContext(text: String) -> String? {
        // Pre-compiled regex patterns for better performance
        let inferencePatterns: [String: [String]] = [
            "steps": [
                "\\b(?:walk|walked|walking)\\b(?!\\s+distance)",
                "\\bhow\\s+much.*(?:walk|walked)\\b",
                "\\bsteps?\\b",
                "\\bpace|paces|stride|strides|footsteps?\\b",
                "\\bmove|moved|movement|activity\\b"
            ],
            "distance": [
                "\\bhow\\s+far\\b",
                "\\b(?:walk|walked|walking)\\s+(?:distance|far)\\b",
                "\\bdistance.*(?:walk|walked)\\b",
                "\\bkilometers?\\b|\\bmiles?\\b|\\bkm\\b",
                "\\btravelled?|traveled|covered|journey|route|path\\b"
            ],
            "heart rate": [
                "\\bheart\\s+rate\\b|\\bheartrate\\b|\\bpulse\\b|\\bhr\\b|\\bbpm\\b",
                "\\bcardiac|cardio|cardiovascular\\b",
                "\\bbeats?\\s+per\\s+minute|rhythm\\b"
            ],
            "calories": [
                "\\bcalories?\\b|\\bkcal\\b|\\benergy\\b|\\bburn\\b",
                "\\bfat\\s+burn|metabolic|metabolism\\b",
                "\\bexpended|consumed|intake\\b"
            ],
            "sleep": [
                "\\bsleep|slept|sleeping|rest|rested\\b",
                "\\bnap|napped|napping|slumber\\b",
                "\\bbedtime|night\\s+sleep|sleep\\s+time\\b"
            ],
            "sleep score": [
                "\\bsleep\\s+(?:quality|score|rating|performance|analysis|efficiency)\\b",
                "\\bhow\\s+well\\s+(?:slept|sleep)\\b"
            ],
            "spo2": [
                "\\boxygen|o2|spo2|saturation|blood\\s+oxygen\\b",
                "\\bpulse\\s+ox|oximeter|breathing\\b"
            ],
            "weight": [
                "\\bweight|weigh|weighing|weighed\\b",
                "\\bbody\\s+mass|bmi|scale\\b",
                "\\bkg|kilogram|pounds?|lbs?\\b"
            ],
            "stress": [
                "\\bstress|stressed|anxiety|anxious\\b",
                "\\btension|worried|overwhelmed\\b",
                "\\bmental\\s+health|relaxation|calm\\b"
            ]
        ]
        
        // Sort by specificity (more specific patterns first)
        let sortedMetrics = inferencePatterns.sorted { $0.value.joined().count > $1.value.joined().count }
        
        for (metric, patterns) in sortedMetrics {
            for pattern in patterns {
                if text.range(of: pattern, options: .regularExpression) != nil {
                    return metric
                }
            }
        }
        
        return nil
    }
    
    private func extractNumber(from text: String) -> Double? {
        let pattern = "\\d+(?:\\.\\d+)?"
        if let match = text.range(of: pattern, options: .regularExpression) {
            let numberString = String(text[match])
            return Double(numberString)
        }
        return nil
    }
}