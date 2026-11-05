# E-AWARE End-to-End Test Matrix

**System:** El Niño Adaptive Fisheries Intelligence (E-AWARE)  
**Test date:** 5 November 2026  
**Primary model:** Literature-based Habitat Suitability Index (HSI)  
**Experimental layer:** Coast-matched conditional logistic model  
**Operational notification:** Email  
**Future delivery channel:** GSM/SMS  

## Purpose

This test matrix documents verification of the complete E-AWARE prototype from environmental-data acquisition through habitat assessment, advisory generation and delivery.

The system is designed to fail closed: where required environmental information is unavailable, E-AWARE must not issue a normal fisheries advisory.

| ID | Test | Method | Expected Result | Result |
|---|---|---|---|---|
| E2E-01 | Copernicus authentication | Live system | Valid credentials accepted | PASS |
| E2E-02 | Live environmental data loading | Live system | Temperature, salinity, SST anomaly and oxygen loaded | PASS |
| E2E-03 | Bathymetry integration | Live system | GEBCO depth attached to ocean grid | PASS |
| E2E-04 | Coast-distance integration | Live system | Distance-to-coast values available | PASS |
| E2E-05 | Oxycline processing | Live system | Seabed-constrained oxycline calculated where physically valid | PASS |
| E2E-06 | Primary HSI generation | Live system | HSI calculated for valid ocean cells | PASS |
| E2E-07 | HSI confidence indicator | Dashboard test | Operational confidence displayed without representing fish-presence probability | PASS |
| E2E-08 | Experimental ML separation | Dashboard test | Experimental score remains secondary and does not alter primary HSI | PASS |
| E2E-09 | Fisheries regulatory layer | Dashboard test | Official regulatory status separated from E-AWARE ecological advisory | PASS |
| E2E-10 | Juvenile-protection layer | Dashboard test | Juvenile status remains NOT ASSESSED when biometric data are unavailable | PASS |
| E2E-11 | Suitable-habitat forecast | Live system | HSI-weighted habitat-centroid movement calculated | PASS |
| E2E-12 | Advisory generation | Live system | Current fisheries advisory generated from primary HSI | PASS |
| E2E-13 | Manual email delivery | Live external test | User-entered email receives current advisory | PASS |
| E2E-14 | Subscriber registration | Live Supabase test | New email stored with active=true | PASS |
| E2E-15 | Subscriber removal | Live Supabase test | Unsubscribe changes active=false | PASS |
| E2E-16 | Subscriber reactivation | Live Supabase test | Re-subscription returns active=true | PASS |
| E2E-17 | Automated advisory workflow | GitHub Actions live test | Current advisory generated and delivered to active subscriber | PASS |
| E2E-18 | Missing advisory file | Automated unit test | Delivery stops safely | PASS |
| E2E-19 | No active subscribers | Automated unit test | Workflow exits without sending | PASS |
| E2E-20 | Supabase failure | Simulated unit test | Delivery stops rather than sending from unknown subscriber state | PASS |
| E2E-21 | Gmail authentication failure | Simulated unit test | Delivery stops and failure is raised | PASS |
| E2E-22 | Duplicate daily delivery | Automated unit test | Already-sent subscriber skipped during scheduled operation | PASS |
| E2E-23 | Manual forced delivery | Automated unit test | Manual workflow may deliberately override duplicate protection | PASS |
| E2E-24 | Environmental-data failure | Controlled GitHub Actions test | Temperature failure detected; advisory generation stops | PASS |
| E2E-25 | No advisory during environmental failure | Controlled GitHub Actions test | Subscriber email is not sent | PASS |
| E2E-26 | Recovery after controlled failure | Live GitHub Actions test | Normal workflow succeeds after failure injection is removed | PASS |

## Verified operational workflow

Copernicus Marine environmental data  
→ environmental preprocessing  
→ primary HSI calculation  
→ HSI confidence assessment  
→ fisheries protection/regulatory context  
→ suitable-habitat forecast  
→ advisory generation  
→ Supabase active-subscriber retrieval  
→ Gmail delivery

## Delivery modes

### Presentation / demonstration mode

A user enters an email address and selects **Send Current E-AWARE Advisory**.

The current advisory is delivered immediately.

### Operational mode

A user subscribes to the **Daily Advisory Subscription**.

Their email address is stored in Supabase and marked active.

GitHub Actions runs the E-AWARE operational workflow automatically at:

**04:00 America/Lima (Peru time)**

The current E-AWARE advisory is generated and sent to active subscribers.

## Safety behaviour

E-AWARE does not issue a normal advisory when required live environmental data cannot be loaded.

A controlled failure test at the **Temperature** stage confirmed that:

- the failure stage was identified;
- advisory generation stopped;
- automated delivery did not proceed;
- the workflow returned an explicit failure state.

Following removal of the simulated failure, a normal workflow run completed successfully, demonstrating recovery without changing the operational configuration.

## Important limitations

- E-AWARE identifies higher-suitability habitat zones, not exact fish locations.
- The primary HSI is literature-based and is not a calibrated probability of anchoveta presence.
- Overall HSI confidence remains capped because independent fish-observation validation is not yet available.
- The experimental ML layer is research-only and does not control the operational advisory.
- Official fishing closures and permissions must be verified through PRODUCE.
- Juvenile-protection status requires current biometric data and is not inferred from environmental variables.
- GEBCO depth represents seafloor bathymetry, not predicted fish depth.
- GSM/SMS delivery remains a future field-deployment implementation.

## Test conclusion

The E-AWARE software prototype successfully demonstrated the complete digital operational chain from live environmental-data retrieval to automated fisheries-advisory delivery.

Controlled failure testing also demonstrated that the system fails closed when required environmental data are unavailable.

The prototype is therefore suitable for demonstration and further field-oriented development, subject to the stated biological-validation, regulatory-data and communications limitations.
