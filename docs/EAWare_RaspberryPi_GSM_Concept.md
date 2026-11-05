# E-AWARE Raspberry Pi + GSM Field Deployment Concept

## Purpose

The current E-AWARE prototype demonstrates the complete software workflow from live environmental-data retrieval to automated fisheries-advisory delivery by email.

For field deployment in coastal fishing communities, E-AWARE proposes a Raspberry Pi and GSM-based communications hub to reduce dependence on smartphones and continuous internet access.

This hardware architecture is a proposed future implementation and has not yet been physically constructed or field tested.

## Proposed Architecture

Copernicus Marine / Environmental Data  
→ E-AWARE cloud processing  
→ Primary HSI and habitat forecast  
→ Fisheries advisory generation  
→ Raspberry Pi field hub  
→ GSM modem  
→ SMS advisory to registered fishers

The existing email system remains available as an internet-based delivery channel.

## Proposed Hardware

### Raspberry Pi

The Raspberry Pi acts as the local field communications hub.

Its proposed responsibilities include:

- receiving E-AWARE advisories;
- caching the most recent advisory locally;
- controlling the GSM modem;
- managing queued SMS messages;
- displaying advisory information locally;
- monitoring connection status;
- supporting future local sensors or additional communications equipment.

### GSM / 4G modem

A GSM-capable modem with a local SIM card would provide SMS communications.

The modem would allow the E-AWARE advisory to reach users who have basic mobile phones and do not require mobile-data access.

The cloud system would continue to generate the scientific advisory; the GSM hardware would provide an additional delivery channel.

### Touchscreen

A small touchscreen, for example approximately 7 inches, could provide a local interface at a landing site, cooperative office or community information point.

It could display:

- current HSI class;
- higher-suitability habitat zone;
- forecast habitat movement;
- environmental conditions;
- regulatory warnings;
- system connectivity status;
- time of the most recent successful update.

### Power system

A field system could use:

- rechargeable battery storage;
- solar trickle charging;
- regulated power electronics;
- mains charging where available.

The intention is to maintain communications during short power interruptions and reduce dependence on continuous grid electricity.

### Enclosure

The Raspberry Pi, modem and power electronics would require a weather-resistant enclosure suitable for a coastal environment.

Design considerations include:

- humidity;
- salt exposure;
- dust;
- heat;
- physical impact;
- cable sealing;
- ventilation.

## Offline and Failure Behaviour

The local hub should not generate its own habitat prediction when live environmental data are unavailable.

Instead, it would store the most recent valid advisory with its timestamp.

If the GSM network is temporarily unavailable:

1. the advisory is stored locally;
2. unsent messages are placed in a queue;
3. the modem periodically checks for network availability;
4. queued messages are transmitted after connectivity returns.

Messages should always contain the advisory date so that users can identify stale information.

If E-AWARE cannot obtain the required environmental data, the cloud system should not issue a normal habitat advisory.

This follows the fail-closed behaviour already demonstrated by the software prototype.

## Proposed SMS Content

A field SMS should remain short and easy to understand.

Example:

E-AWARE 05/11/2026

Anchoveta habitat: MODERATE  
HSI: 0.49

Higher-suitability zone:  
-4.5833, -81.5000

Habitat shift: No clear shift

Check official PRODUCE restrictions.

E-AWARE indicates habitat suitability, not exact fish location.

## Relationship to the Current Prototype

### Implemented

- live Copernicus environmental-data retrieval;
- literature-based primary HSI;
- habitat forecast;
- HSI confidence indicator;
- fisheries-protection advisory layer;
- manual email delivery;
- persistent user subscription;
- automated daily email delivery;
- failure-mode protection;
- scheduled cloud operation.

### Future implementation

- Raspberry Pi field communications hub;
- GSM/SMS modem;
- local touchscreen;
- local offline advisory cache;
- SMS transmission queue;
- solar/battery power system;
- weather-resistant enclosure;
- coastal field trials.

## Why GSM/SMS is Proposed

SMS provides a practical future communications channel because it does not require a smartphone application or continuous mobile-data access.

This could improve accessibility for users operating in areas where internet connectivity is limited while allowing E-AWARE's scientific computation to remain in the cloud.

## Engineering Boundary

The Raspberry Pi and GSM system would be a communications and resilience layer.

It would not replace the E-AWARE habitat model.

Environmental processing, HSI calculation, forecast generation and regulatory safeguards would remain controlled by the validated cloud workflow.

## Future Validation

Before real deployment, the hardware system would require testing of:

- GSM coverage;
- SMS delivery reliability;
- power consumption;
- battery endurance;
- solar charging performance;
- enclosure protection;
- local operating temperature;
- offline message queuing;
- message latency;
- user comprehension;
- maintenance requirements.

## Conclusion

The Raspberry Pi + GSM architecture provides a realistic pathway from the current E-AWARE software prototype toward a field-deployable fisheries-information system.

The current email implementation demonstrates that the complete digital advisory workflow functions successfully.

The proposed hardware layer would extend that same advisory to users through SMS and local community infrastructure without changing the primary scientific model.
