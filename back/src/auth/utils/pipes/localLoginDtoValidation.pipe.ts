import { PipeTransform, Injectable, ArgumentMetadata, BadRequestException, HttpException, HttpStatus } from '@nestjs/common';
import { LocalLoginDto } from "../../dto/localLoginDto";

@Injectable()
export class LocalLoginDtoValidationPipe implements PipeTransform<any> {
  transform(value: any, metadata: ArgumentMetadata) {
    const { email, password } = value as LocalLoginDto;

    if (!email || !password) {
      throw new HttpException("등록되지 않은 이메일 이거나, 유효하지 않은 비밀번호입니다.", HttpStatus.UNAUTHORIZED);
    }

    return value;
  }
}