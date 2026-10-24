"""Read-only real landing capture plus the validated200Hz contact instrumentation.

Python MRO joins the two existing step/reset/close hooks; no model/action/reset
changes during collection. Dataset success never substitutes source-start success.
"""
import evaluate_jump
from capture_landing_states import CaptureEnv
from evaluate_contact_instrumented_jump import ContactInstrumentedEnv

class ContactCaptureEnv(CaptureEnv, ContactInstrumentedEnv):
    pass

if __name__ == '__main__':
    evaluate_jump.JumpEnv = ContactCaptureEnv
    evaluate_jump.main()
