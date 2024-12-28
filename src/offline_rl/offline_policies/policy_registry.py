from enum import Enum, StrEnum

from offline_rl.offline_policies.bcq_continuous_policy import BCQContinuousPolicyModel
from offline_rl.offline_policies.bcq_discrete_policy import BCQDiscretePolicyModel
from offline_rl.offline_policies.cql_continuous_policy import CQLContinuousPolicyModel
from offline_rl.offline_policies.cql_discrete_policy import CQLDiscretePolicyModel
from offline_rl.offline_policies.dqn_policy import DQNPolicyModel
from offline_rl.offline_policies.il_policy import ILPolicyModel
from offline_rl.offline_policies.ppo_policy import PpoPolicyModel


class CallableEnum(Enum):
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class PolicyType(str, Enum):
    offline = "offline"
    onpolicy = "onpolicy"
    offpolicy = "offpolicy"


class RLPolicies(StrEnum):
    bcq_discrete = "bcq_discrete"
    cql_continuous = "cql_continuous"
    imitation_learning = "imitation_learning"
    bcq_continuous = "bcq_continuous"
    cql_discrete = "cql_discrete"
    dqn = "dqn"
    ppo = "ppo"


class RLPolicyFactory(StrEnum):
    bcq_discrete = RLPolicies.bcq_discrete
    cql_continuous = RLPolicies.cql_continuous
    imitation_learning = RLPolicies.imitation_learning
    bcq_continuous = RLPolicies.bcq_continuous
    cql_discrete = RLPolicies.cql_discrete
    dqn = RLPolicies.dqn
    ppo = RLPolicies.ppo

    def get_policy(self):
        match self:
            case self.bcq_discrete:
                return BCQDiscretePolicyModel()
            case self.bcq_continuous:
                return BCQContinuousPolicyModel()
            case self.cql_continuous:
                return CQLContinuousPolicyModel()
            case self.cql_discrete:
                return CQLDiscretePolicyModel()
            case self.dqn:
                return DQNPolicyModel()
            case self.imitation_learning:
                return ILPolicyModel()
            case self.ppo:
                return PpoPolicyModel()
