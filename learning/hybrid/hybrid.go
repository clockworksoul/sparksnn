// Package hybrid implements a combined learning rule that runs
// reward-modulated STDP and weight perturbation simultaneously.
//
// The two mechanisms operate on different timescales:
//
//   - R-STDP provides fine-grained, spike-timing-based credit
//     assignment. It strengthens connections that causally contribute
//     to correct output spikes and weakens anti-causal ones.
//
//   - Weight perturbation provides coarse, gradient-free exploration.
//     It randomly tweaks individual weights and keeps changes that
//     improve overall performance.
//
// Together, perturbation handles global search (escaping local
// optima, activating dead neurons) while R-STDP handles local
// refinement (sharpening timing-dependent patterns).
//
// This mirrors biological learning where multiple plasticity
// mechanisms coexist: Hebbian/STDP for local synaptic tuning,
// neuromodulatory signals for global reinforcement, and synaptic
// noise for exploration.
package hybrid

import (
	bio "github.com/clockworksoul/sparksnn"
	"github.com/clockworksoul/sparksnn/learning/perturbation"
	"github.com/clockworksoul/sparksnn/learning/rstdp"
)

// Config holds configuration for the hybrid learning rule.
type Config struct {
	// RSTDP is the configuration for the R-STDP component.
	RSTDP rstdp.Config

	// Perturbation is the configuration for the weight perturbation
	// component.
	Perturbation perturbation.Config
}

// DefaultConfig returns reasonable defaults for hybrid learning.
func DefaultConfig() Config {
	rc := rstdp.DefaultConfig()
	rc.APlus = 50              // gentler than standalone R-STDP
	rc.AMinus = 50             // to avoid fighting perturbation
	rc.MaxWeightMagnitude = 5000

	pc := perturbation.DefaultConfig()
	pc.PerturbSize = 200       // smaller perturbations alongside STDP
	pc.MaxPerturbSize = 2000
	pc.MaxWeightMagnitude = 5000

	return Config{
		RSTDP:        rc,
		Perturbation: pc,
	}
}

// Rule implements the hybrid R-STDP + perturbation learning rule.
type Rule struct {
	rstdpRule  *rstdp.Rule
	perturbRule *perturbation.Rule
}

// NewRule creates a hybrid learning rule from the given config.
func NewRule(config Config) *Rule {
	return &Rule{
		rstdpRule:   rstdp.NewRule(config.RSTDP),
		perturbRule: perturbation.NewRule(config.Perturbation),
	}
}

// OnSpikePropagation delegates to R-STDP for eligibility trace
// updates based on pre→post spike timing.
// Perturbation does not use spike timing — no-op on its side.
func (r *Rule) OnSpikePropagation(conn *bio.Connection, preFiredAt, postLastFired uint32) {
	r.rstdpRule.OnSpikePropagation(conn, preFiredAt, postLastFired)
}

// OnPostFire delegates to R-STDP for eligibility trace updates
// based on post-synaptic firing and incoming spike timing.
func (r *Rule) OnPostFire(incoming []bio.IncomingConnection, postFiredAt uint32) {
	r.rstdpRule.OnPostFire(incoming, postFiredAt)
}

// OnReward delegates to both learning rules:
//   - R-STDP consolidates eligibility traces into weight changes.
//   - Perturbation evaluates its pending perturbation and applies
//     a new one.
//
// R-STDP runs first so its fine-grained adjustments are in place
// before perturbation evaluates network performance.
func (r *Rule) OnReward(net *bio.Network, reward int32, tick uint32) {
	r.rstdpRule.OnReward(net, reward, tick)
	r.perturbRule.OnReward(net, reward, tick)
}

// Maintain delegates to R-STDP for eligibility trace decay.
// Perturbation has no per-tick maintenance.
func (r *Rule) Maintain(net *bio.Network, tick uint32) {
	r.rstdpRule.Maintain(net, tick)
}
