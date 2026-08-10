import pytest

from regrag.metadata import parse_source_filename


def test_parses_country_agency_and_norm_from_filename():
    meta = parse_source_filename("AR_SIC_res-16-2025_texto.html")

    assert meta.country == "AR"
    assert meta.agency == "SIC"
    assert meta.norm_id == "res-16-2025"


def test_captures_variant_so_two_renderings_of_one_norm_stay_distinct():
    boletin = parse_source_filename("AR_SIC_res-16-2025_boletin.html")
    texto = parse_source_filename("AR_SIC_res-16-2025_texto.html")

    assert boletin.variant == "boletin"
    assert texto.variant == "texto"
    assert boletin.norm_id == texto.norm_id


def test_variant_is_none_when_filename_has_no_fourth_part():
    meta = parse_source_filename("PY_INTN_direccion-seguridad-electrica.html")

    assert meta.norm_id == "direccion-seguridad-electrica"
    assert meta.variant is None


def test_multipart_variant_is_kept_whole():
    meta = parse_source_filename("CL_SEC_rex-28201_protocolo-pe-8-11.pdf")

    assert meta.norm_id == "rex-28201"
    assert meta.variant == "protocolo-pe-8-11"


def test_extracts_four_digit_year_from_norm_id():
    assert parse_source_filename("AR_SIC_res-16-2025_texto.html").year == 2025
    assert parse_source_filename("PE_INDECOPI_ntp-370-053-1999.pdf").year == 1999


def test_year_is_none_when_norm_id_carries_no_year():
    assert parse_source_filename("CO_MINENERGIA_res-9703.pdf").year is None


def test_rejects_filename_that_breaks_the_naming_convention():
    with pytest.raises(ValueError, match="naming convention"):
        parse_source_filename("random-download.pdf")
