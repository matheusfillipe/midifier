"""Key checking. The service stores a digest, never the key."""

from midifier.auth import generate_key
from midifier.auth import hash_key
from midifier.auth import verify


class TestGenerateKey:
    def test_keys_are_unique(self) -> None:
        assert generate_key() != generate_key()

    def test_keys_carry_real_entropy(self) -> None:
        """The digest is only safe because guessing the key is infeasible."""
        assert len(generate_key()) >= 32


class TestHashKey:
    def test_is_stable(self) -> None:
        assert hash_key("abc") == hash_key("abc")

    def test_differs_per_key(self) -> None:
        assert hash_key("abc") != hash_key("abd")

    def test_does_not_contain_the_key(self) -> None:
        """Whoever can read the deployed hash still cannot call the service."""
        key = generate_key()
        assert key not in hash_key(key)


class TestVerify:
    def test_accepts_the_right_key(self) -> None:
        key = generate_key()
        assert verify(key, hash_key(key)) is True

    def test_refuses_the_wrong_key(self) -> None:
        assert verify("wrong", hash_key(generate_key())) is False

    def test_refuses_a_missing_key_when_one_is_required(self) -> None:
        assert verify(None, hash_key("secret")) is False
        assert verify("", hash_key("secret")) is False

    def test_open_when_no_hash_is_configured(self) -> None:
        """A local run with no key set should need no credentials."""
        assert verify(None, None) is True
        assert verify("anything", None) is True

    def test_a_hash_is_not_accepted_as_the_key(self) -> None:
        """Presenting the stored digest must not authenticate."""
        digest = hash_key("secret")
        assert verify(digest, digest) is False
